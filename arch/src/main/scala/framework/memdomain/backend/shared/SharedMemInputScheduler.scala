package framework.memdomain.backend.shared

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.top.GlobalConfig

@instantiable
class SharedMemInputScheduler(val b: GlobalConfig, val issuePortsK: Int) extends Module {
  private val totalBanks   = SharedMemLayout.totalBank(b)
  private val totalChannel = SharedMemLayout.totalChannel(b)
  private val pbankWidth   = math.max(1, log2Ceil(totalBanks))
  private val srcWidth     = math.max(1, log2Ceil(totalChannel))
  private val chanCountW   = math.max(1, log2Ceil(totalChannel + 1))
  private val issueCountW  = math.max(1, log2Ceil(issuePortsK + 1))

  require(totalChannel > 0, "SharedMemInputScheduler requires at least one input channel")
  require(issuePortsK > 0, "SharedMemInputScheduler requires at least one issue port")
  require(issuePortsK <= totalChannel, s"issuePortsK($issuePortsK) must be <= totalChannel($totalChannel)")

  @public
  val io = IO(new Bundle {
    val inRead  = Vec(totalChannel, new SramReadIO(b))
    val inWrite = Vec(totalChannel, new SramWriteIO(b))

    val targetValid = Input(Vec(totalChannel, Bool()))
    val targetPbank = Input(Vec(totalChannel, UInt(pbankWidth.W)))

    val bankRead  = Vec(totalBanks, Flipped(new SramReadIO(b)))
    val bankWrite = Vec(totalBanks, Flipped(new SramWriteIO(b)))

    val readIssued  = Output(Vec(totalChannel, Bool()))
    val writeIssued = Output(Vec(totalChannel, Bool()))
    val issuedPbank = Output(Vec(totalChannel, UInt(pbankWidth.W)))

    val activeInputCount       = Output(UInt(chanCountW.W))
    val issuedCount            = Output(UInt(issueCountW.W))
    val bankConflictStallCount = Output(UInt(chanCountW.W))
    val noIssuePortStallCount  = Output(UInt(chanCountW.W))
  })

  private def orReduce(xs: Seq[Bool]): Bool =
    xs.reduceOption(_ || _).getOrElse(false.B)

  private def wrapChannel(idx: UInt): UInt = {
    val wrapped = Mux(idx >= totalChannel.U, idx - totalChannel.U, idx)
    wrapped(srcWidth - 1, 0)
  }

  val candidateValid   = Wire(Vec(totalChannel, Bool()))
  val candidateIsWrite = Wire(Vec(totalChannel, Bool()))
  val readReqBits      = Wire(Vec(totalChannel, chiselTypeOf(io.inRead(0).req.bits)))
  val writeReqBits     = Wire(Vec(totalChannel, chiselTypeOf(io.inWrite(0).req.bits)))

  for (ch <- 0 until totalChannel) {
    candidateValid(ch)   := io.inWrite(ch).req.valid || io.inRead(ch).req.valid
    candidateIsWrite(ch) := io.inWrite(ch).req.valid
    readReqBits(ch)      := io.inRead(ch).req.bits
    writeReqBits(ch)     := io.inWrite(ch).req.bits

    when(io.inWrite(ch).req.valid && io.inRead(ch).req.valid) {
      assert(false.B, "SharedMemInputScheduler channel has simultaneous read and write request: ch=%d\n", ch.U)
    }
  }

  val issueValid     = Wire(Vec(issuePortsK, Bool()))
  val issueSrc       = Wire(Vec(issuePortsK, UInt(srcWidth.W)))
  val issuePbank     = Wire(Vec(issuePortsK, UInt(pbankWidth.W)))
  val issueIsWrite   = Wire(Vec(issuePortsK, Bool()))
  val issueReadBits  = Wire(Vec(issuePortsK, chiselTypeOf(io.inRead(0).req.bits)))
  val issueWriteBits = Wire(Vec(issuePortsK, chiselTypeOf(io.inWrite(0).req.bits)))

  val rrPtr = RegInit(0.U(srcWidth.W))

  for (slot <- 0 until issuePortsK) {
    val eligibleByOrder = Wire(Vec(totalChannel, Bool()))

    for (pos <- 0 until totalChannel) {
      val idx = wrapChannel(rrPtr + pos.U)
      val channelAlreadyIssued = orReduce((0 until slot).map(prev => issueValid(prev) && issueSrc(prev) === idx))
      val bankAlreadyIssued = orReduce(
        (0 until slot).map(prev => issueValid(prev) && issuePbank(prev) === io.targetPbank(idx))
      )

      eligibleByOrder(pos) :=
        candidateValid(idx) &&
          io.targetValid(idx) &&
          !channelAlreadyIssued &&
          !bankAlreadyIssued
    }

    val selectedPos = PriorityEncoder(eligibleByOrder)
    val selectedSrc = wrapChannel(rrPtr + selectedPos)

    issueValid(slot)     := eligibleByOrder.asUInt.orR
    issueSrc(slot)       := selectedSrc
    issuePbank(slot)     := io.targetPbank(selectedSrc)
    issueIsWrite(slot)   := candidateIsWrite(selectedSrc)
    issueReadBits(slot)  := readReqBits(selectedSrc)
    issueWriteBits(slot) := writeReqBits(selectedSrc)
  }

  val anyIssue  = issueValid.asUInt.orR
  val lastGrant = Wire(UInt(srcWidth.W))
  lastGrant := issueSrc(0)
  for (slot <- 0 until issuePortsK) {
    when(issueValid(slot)) {
      lastGrant := issueSrc(slot)
    }
  }

  when(anyIssue) {
    rrPtr := wrapChannel(lastGrant + 1.U)
  }

  val grantVec          = Wire(Vec(totalChannel, Bool()))
  val bankConflictStall = Wire(Vec(totalChannel, Bool()))
  val noIssuePortStall  = Wire(Vec(totalChannel, Bool()))

  for (ch <- 0 until totalChannel) {
    grantVec(ch) := orReduce((0 until issuePortsK).map(slot => issueValid(slot) && issueSrc(slot) === ch.U))

    val conflictsWithGrantedBank =
      orReduce((0 until issuePortsK).map(slot => issueValid(slot) && issuePbank(slot) === io.targetPbank(ch)))

    bankConflictStall(ch) :=
      candidateValid(ch) &&
        io.targetValid(ch) &&
        !grantVec(ch) &&
        conflictsWithGrantedBank

    noIssuePortStall(ch) :=
      candidateValid(ch) &&
        io.targetValid(ch) &&
        !grantVec(ch) &&
        !bankConflictStall(ch) &&
        PopCount(issueValid) === issuePortsK.U
  }

  io.activeInputCount       := PopCount(candidateValid)
  io.issuedCount            := PopCount(issueValid)
  io.bankConflictStallCount := PopCount(bankConflictStall)
  io.noIssuePortStallCount  := PopCount(noIssuePortStall)

  for (ch <- 0 until totalChannel) {
    io.inRead(ch).req.ready  := grantVec(ch) && !candidateIsWrite(ch)
    io.inWrite(ch).req.ready := grantVec(ch) && candidateIsWrite(ch)

    io.readIssued(ch)  := io.inRead(ch).req.fire
    io.writeIssued(ch) := io.inWrite(ch).req.fire
    io.issuedPbank(ch) := io.targetPbank(ch)

    io.inRead(ch).resp.valid     := false.B
    io.inRead(ch).resp.bits.data := DontCare

    io.inWrite(ch).resp.valid   := false.B
    io.inWrite(ch).resp.bits.ok := DontCare
  }

  for (bank <- 0 until totalBanks) {
    io.bankRead(bank).req.valid  := false.B
    io.bankRead(bank).req.bits   := DontCare
    io.bankRead(bank).resp.ready := true.B

    io.bankWrite(bank).req.valid  := false.B
    io.bankWrite(bank).req.bits   := DontCare
    io.bankWrite(bank).resp.ready := true.B
  }

  val readRespSrc  = RegInit(VecInit(Seq.fill(totalBanks)(0.U(srcWidth.W))))
  val writeRespSrc = RegInit(VecInit(Seq.fill(totalBanks)(0.U(srcWidth.W))))

  for (bank <- 0 until totalBanks) {
    val readHits = Wire(Vec(issuePortsK, Bool()))
    val writeHits = Wire(Vec(issuePortsK, Bool()))

    for (slot <- 0 until issuePortsK) {
      readHits(slot) :=
        issueValid(slot) &&
          !issueIsWrite(slot) &&
          issuePbank(slot) === bank.U

      writeHits(slot) :=
        issueValid(slot) &&
          issueIsWrite(slot) &&
          issuePbank(slot) === bank.U
    }

    io.bankRead(bank).req.valid := readHits.asUInt.orR
    io.bankRead(bank).req.bits  := Mux1H((0 until issuePortsK).map(slot => readHits(slot) -> issueReadBits(slot)))

    io.bankWrite(bank).req.valid := writeHits.asUInt.orR
    io.bankWrite(bank).req.bits  := Mux1H((0 until issuePortsK).map(slot => writeHits(slot) -> issueWriteBits(slot)))

    when(io.bankRead(bank).req.fire) {
      readRespSrc(bank) := Mux1H((0 until issuePortsK).map(slot => readHits(slot) -> issueSrc(slot)))
    }

    when(io.bankWrite(bank).req.fire) {
      writeRespSrc(bank) := Mux1H((0 until issuePortsK).map(slot => writeHits(slot) -> issueSrc(slot)))
    }
  }

  for (ch <- 0 until totalChannel) {
    val readRespHits = Wire(Vec(totalBanks, Bool()))
    val writeRespHits = Wire(Vec(totalBanks, Bool()))

    for (bank <- 0 until totalBanks) {
      readRespHits(bank)  := io.bankRead(bank).resp.valid && readRespSrc(bank) === ch.U
      writeRespHits(bank) := io.bankWrite(bank).resp.valid && writeRespSrc(bank) === ch.U
    }

    io.inRead(ch).resp.valid     := readRespHits.asUInt.orR
    io.inRead(ch).resp.bits.data := Mux1H(
      (0 until totalBanks).map(bank => readRespHits(bank) -> io.bankRead(bank).resp.bits.data)
    )

    io.inWrite(ch).resp.valid   := writeRespHits.asUInt.orR
    io.inWrite(ch).resp.bits.ok := Mux1H(
      (0 until totalBanks).map(bank => writeRespHits(bank) -> io.bankWrite(bank).resp.bits.ok)
    )
  }

  private val enableRouterCounters = sys.env.get("BB_SHARED_ROUTER_COUNTERS").contains("1")
  if (enableRouterCounters) {
    val totalActiveInputs      = RegInit(0.U(64.W))
    val totalIssued           = RegInit(0.U(64.W))
    val totalBankConflict     = RegInit(0.U(64.W))
    val totalNoIssuePortStall = RegInit(0.U(64.W))
    val perChannelGrant       = RegInit(VecInit(Seq.fill(totalChannel)(0.U(64.W))))
    val perBankIssue          = RegInit(VecInit(Seq.fill(totalBanks)(0.U(64.W))))

    totalActiveInputs      := totalActiveInputs + io.activeInputCount
    totalIssued           := totalIssued + io.issuedCount
    totalBankConflict     := totalBankConflict + io.bankConflictStallCount
    totalNoIssuePortStall := totalNoIssuePortStall + io.noIssuePortStallCount

    for (ch <- 0 until totalChannel) {
      when(grantVec(ch)) {
        perChannelGrant(ch) := perChannelGrant(ch) + 1.U
      }
    }

    for (bank <- 0 until totalBanks) {
      when(io.bankRead(bank).req.fire || io.bankWrite(bank).req.fire) {
        perBankIssue(bank) := perBankIssue(bank) + 1.U
      }
    }

    dontTouch(totalActiveInputs)
    dontTouch(totalIssued)
    dontTouch(totalBankConflict)
    dontTouch(totalNoIssuePortStall)
    dontTouch(perChannelGrant)
    dontTouch(perBankIssue)
  }
}
