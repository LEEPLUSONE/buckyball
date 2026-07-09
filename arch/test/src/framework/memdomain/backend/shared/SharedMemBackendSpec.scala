package framework.memdomain.backend.shared

import chisel3._
import chiseltest._
import framework.memdomain.configs.MemDomainParam
import framework.top.GlobalConfig
import org.scalatest.flatspec.AnyFlatSpec

class SharedMemBackendSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SharedMemBackend"

  private val cores    = 4
  private val channels = 4
  private val banks    = 4

  private def backendConfig: GlobalConfig = {
    val base = GlobalConfig()
    base.copy(
      memDomain = MemDomainParam(
        bankNum = banks,
        bankWidth = 128,
        bankEntries = 16,
        bankMaskLen = 16,
        sharedEnable = true,
        sharedEntries = 16,
        sharedInputChannels = channels,
        sharedDefaultGroupCount = 1,
        tlb_size = 4,
        dma_n_xacts = 4,
        dma_burst_maxbytes = 64,
        bankChannel = 1,
        max_in_flight_mem_reqs = 4,
        dma_buswidth = 128,
        memAddrLen = 32,
        tmaReadChannel = 1,
        tmaWriteChannel = 1,
        mmioBankNum = 1,
        mmioBankEntries = 16,
        mmioBankWidth = 32,
        mmioReadWidth = 32
      ),
      top = base.top.copy(nCores = cores)
    )
  }

  private def init(c: SharedMemBackend): Unit = {
    c.io.config.valid.poke(false.B)
    c.io.config.bits.vbank_id.poke(0.U)
    c.io.config.bits.is_shared.poke(true.B)
    c.io.config.bits.is_multi.poke(false.B)
    c.io.config.bits.alloc.poke(true.B)
    c.io.config.bits.group_id.poke(0.U)
    c.io.config.bits.hart_id.poke(0.U)

    for (q <- 0 until cores) {
      c.io.query_valid(q).poke(false.B)
      c.io.query_hart_id(q).poke(0.U)
      c.io.query_vbank_id(q).poke(0.U)
    }

    for (ch <- 0 until channels) {
      c.io.mem_req(ch).hart_id.poke(0.U)
      c.io.mem_req(ch).bank_id.poke(0.U)
      c.io.mem_req(ch).group_id.poke(0.U)
      c.io.mem_req(ch).is_shared.poke(true.B)

      c.io.mem_req(ch).read.req.valid.poke(false.B)
      c.io.mem_req(ch).read.req.bits.addr.poke(0.U)
      c.io.mem_req(ch).read.resp.ready.poke(true.B)

      c.io.mem_req(ch).write.req.valid.poke(false.B)
      c.io.mem_req(ch).write.req.bits.addr.poke(0.U)
      c.io.mem_req(ch).write.req.bits.data.poke(0.U)
      c.io.mem_req(ch).write.req.bits.mask.foreach(_.poke(false.B))
      c.io.mem_req(ch).write.resp.ready.poke(true.B)
    }
  }

  private def allocate(c: SharedMemBackend, hart: Int, vbank: Int, group: Int = 0): Unit = {
    c.io.config.valid.poke(true.B)
    c.io.config.bits.alloc.poke(true.B)
    c.io.config.bits.is_shared.poke(true.B)
    c.io.config.bits.is_multi.poke(false.B)
    c.io.config.bits.hart_id.poke(hart.U)
    c.io.config.bits.vbank_id.poke(vbank.U)
    c.io.config.bits.group_id.poke(group.U)
    c.io.config.ready.expect(true.B)
    c.clock.step()
    c.io.config.valid.poke(false.B)
  }

  private def driveWrite(c: SharedMemBackend, ch: Int, hart: Int, vbank: Int, data: Int): Unit = {
    c.io.mem_req(ch).hart_id.poke(hart.U)
    c.io.mem_req(ch).bank_id.poke(vbank.U)
    c.io.mem_req(ch).group_id.poke(0.U)
    c.io.mem_req(ch).is_shared.poke(true.B)
    c.io.mem_req(ch).write.req.valid.poke(true.B)
    c.io.mem_req(ch).write.req.bits.addr.poke(ch.U)
    c.io.mem_req(ch).write.req.bits.data.poke(data.U)
    c.io.mem_req(ch).write.req.bits.mask.foreach(_.poke(true.B))
  }

  it should "allocate distinct physical banks for independent shared vbank mappings" in {
    test(new SharedMemBackend(backendConfig)) { c =>
      init(c)

      for (hart <- 0 until cores) {
        allocate(c, hart = hart, vbank = hart)
      }

      for (ch <- 0 until channels) {
        driveWrite(c, ch = ch, hart = ch, vbank = ch, data = ch + 1)
      }

      for (ch <- 0 until channels) {
        c.io.mem_req(ch).write.req.ready.expect(true.B)
      }

      c.clock.step()

      for (ch <- 0 until channels) {
        c.io.mem_req(ch).write.req.valid.poke(false.B)
      }

      c.clock.step()

      for (ch <- 0 until channels) {
        c.io.mem_req(ch).write.resp.valid.expect(true.B)
        c.io.mem_req(ch).write.resp.bits.ok.expect(true.B)
      }
    }
  }
}
