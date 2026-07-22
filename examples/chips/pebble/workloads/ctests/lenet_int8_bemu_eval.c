#define main lenet_single_image_main
#include "lenet_int8_bemu.c"
#undef main

#define LENET_EVAL_VERSION 2u

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t input_count;
  uint32_t layer_count;
  uint32_t sample_count;
} EvalPayloadHeader;

typedef struct {
  uint32_t label;
  uint32_t fp32_class;
  uint32_t quant_class;
  int8_t expected_q[TILE];
} EvalSampleHeader;

_Static_assert(sizeof(EvalPayloadHeader) == 20,
               "evaluation payload header layout changed");
_Static_assert(sizeof(EvalSampleHeader) == 28,
               "evaluation sample header layout changed");

static FILE *load_eval_payload(const char *path, EvalPayloadHeader *payload,
                               Layer layers[LAYER_COUNT]) {
  FILE *stream = fopen(path, "rb");
  if (stream == NULL) {
    fprintf(stderr, "[LeNet/Pebble] cannot open payload: %s\n", path);
    exit(1);
  }
  checked_read(payload, sizeof(*payload), 1, stream, "evaluation header");
  if (payload->magic != LENET_MAGIC || payload->version != LENET_EVAL_VERSION ||
      payload->input_count != 28 * 28 || payload->layer_count != LAYER_COUNT ||
      payload->sample_count == 0)
    die("invalid evaluation payload header");

  for (int index = 0; index < LAYER_COUNT; ++index) {
    Layer *layer = &layers[index];
    checked_read(&layer->header, sizeof(layer->header), 1, stream,
                 "layer header");
    if (layer->header.weight_count !=
            layer->header.k * layer->header.n_padded ||
        layer->header.bias_count != layer->header.n ||
        layer->header.n_padded % TILE != 0)
      die("invalid layer dimensions in evaluation payload");

    float *weight = (float *)checked_malloc((size_t)layer->header.weight_count *
                                            sizeof(float));
    layer->weight =
        (int8_t *)checked_malloc((size_t)layer->header.weight_count);
    layer->bias = (int32_t *)checked_malloc((size_t)layer->header.bias_count *
                                            sizeof(int32_t));
    checked_read(weight, sizeof(float), layer->header.weight_count, stream,
                 "layer weight");
    checked_read(layer->bias, sizeof(int32_t), layer->header.bias_count, stream,
                 "layer bias");
    pebble_fp2int(weight, layer->weight, layer->header.weight_count,
                  layer->header.weight_multiplier_bits);
    free(weight);
  }
  return stream;
}

static int8_t *run_eval_inference(const float *input_fp32,
                                  Layer layers[LAYER_COUNT]) {
  int8_t *qinput = (int8_t *)checked_malloc(28 * 28);
  pebble_fp2int(input_fp32, qinput, 28 * 28,
                layers[0].header.input_multiplier_bits);

  int8_t *conv1 = run_conv(qinput, 28, 28, 1, &layers[0]);
  free(qinput);
  relu_int8(conv1, 24 * 24 * 6);
  int8_t *pool1 = maxpool2(conv1, 24, 24, 6);
  free(conv1);

  int8_t *conv2_input = rescale_with_padding(
      pool1, 12 * 12 * 6, layers[0].header.output_step_bits,
      layers[1].header.input_multiplier_bits);
  free(pool1);
  int8_t *conv2 = run_conv(conv2_input, 12, 12, 6, &layers[1]);
  free(conv2_input);
  relu_int8(conv2, 8 * 8 * 16);
  int8_t *pool2 = maxpool2(conv2, 8, 8, 16);
  free(conv2);

  int8_t *fc1_nhwc =
      rescale_with_padding(pool2, 4 * 4 * 16, layers[1].header.output_step_bits,
                           layers[2].header.input_multiplier_bits);
  free(pool2);
  int8_t *fc1_input = flatten_nchw(fc1_nhwc, 4, 4, 16);
  free(fc1_nhwc);
  int8_t *fc1 = run_linear(fc1_input, &layers[2]);
  free(fc1_input);
  relu_int8(fc1, 120);

  int8_t *fc2_input =
      rescale_with_padding(fc1, 120, layers[2].header.output_step_bits,
                           layers[3].header.input_multiplier_bits);
  free(fc1);
  int8_t *fc2 = run_linear(fc2_input, &layers[3]);
  free(fc2_input);
  relu_int8(fc2, 84);

  int8_t *fc3_input =
      rescale_with_padding(fc2, 84, layers[3].header.output_step_bits,
                           layers[4].header.input_multiplier_bits);
  free(fc2);
  int8_t *fc3 = run_linear(fc3_input, &layers[4]);
  free(fc3_input);
  return fc3;
}

int main(void) {
  const char *payload_path = "lenet_int8_eval_payload.bin";
  EvalPayloadHeader payload;
  Layer layers[LAYER_COUNT];
  memset(layers, 0, sizeof(layers));

  printf("[LeNet/Pebble] loading evaluation payload %s\n", payload_path);
  FILE *stream = load_eval_payload(payload_path, &payload, layers);
  float *input =
      (float *)checked_malloc((size_t)payload.input_count * sizeof(float));
  unsigned long fp32_correct = 0;
  unsigned long int8_correct = 0;
  unsigned long prediction_agreement = 0;
  unsigned long python_quant_agreement = 0;

  for (uint32_t sample = 0; sample < payload.sample_count; ++sample) {
    EvalSampleHeader expected;
    checked_read(&expected, sizeof(expected), 1, stream, "sample header");
    checked_read(input, sizeof(float), payload.input_count, stream,
                 "sample input");
    int8_t *q = run_eval_inference(input, layers);
    int classification = 0;
    for (int i = 1; i < 10; ++i)
      if (q[i] > q[classification])
        classification = i;

    if ((uint32_t)classification == expected.label)
      ++int8_correct;
    if (expected.fp32_class == expected.label)
      ++fp32_correct;
    if ((uint32_t)classification == expected.fp32_class)
      ++prediction_agreement;
    if ((uint32_t)classification == expected.quant_class &&
        memcmp(q, expected.expected_q, 10) == 0)
      ++python_quant_agreement;
    else {
      printf("[LeNet/Pebble] Python mismatch sample=%u class=%d expected=%u\n",
             sample, classification, expected.quant_class);
      free(q);
      die("Pebble BEMU differs from Python INT8 reference");
    }
    free(q);
    verify_numerics = 0;
    if ((sample + 1) % 1000 == 0 || sample + 1 == payload.sample_count)
      printf("[LeNet/Pebble] progress=%u/%u int8_correct=%lu\n", sample + 1,
             payload.sample_count, int8_correct);
  }
  if (fgetc(stream) != EOF)
    die("evaluation payload has unexpected trailing data");
  fclose(stream);

  double samples = (double)payload.sample_count;
  printf("[LeNet/Pebble] samples=%u\n", payload.sample_count);
  printf("[LeNet/Pebble] FP32 Top1=%lu/%u %.4f%%\n", fp32_correct,
         payload.sample_count, 100.0 * fp32_correct / samples);
  printf("[LeNet/Pebble] Pebble INT8 Top1=%lu/%u %.4f%%\n", int8_correct,
         payload.sample_count, 100.0 * int8_correct / samples);
  printf("[LeNet/Pebble] Top1 loss=%.4f percentage-points\n",
         100.0 * ((double)fp32_correct - (double)int8_correct) / samples);
  printf("[LeNet/Pebble] FP32 prediction agreement=%lu/%u %.4f%%\n",
         prediction_agreement, payload.sample_count,
         100.0 * prediction_agreement / samples);
  printf("[LeNet/Pebble] Python INT8 exact agreement=%lu/%u %.4f%%\n",
         python_quant_agreement, payload.sample_count,
         100.0 * python_quant_agreement / samples);
  printf("[LeNet/Pebble] issues: FP2INT=%lu INT2FP/requant=%lu MATRIX=%lu\n",
         fp2int_issues, int2fp_issues, matrix_issues);

  free(input);
  for (int index = 0; index < LAYER_COUNT; ++index) {
    free(layers[index].weight);
    free(layers[index].bias);
  }
  return 0;
}
