package framework.memdomain.backend.shared

import chisel3._
import chiseltest._
import framework.memdomain.configs.MemDomainParam
import framework.top.GlobalConfig
import org.scalatest.flatspec.AnyFlatSpec

class SharedMemInputSchedulerSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SharedMemInputScheduler"

  private val channels = 4
  private val banks    = 4
  private val issueK   = 2

  private def schedulerConfig(channelCount: Int = channels, bankCount: Int = banks): GlobalConfig = {
    val base = GlobalConfig()
    base.copy(
      memDomain = MemDomainParam(
        bankNum = bankCount,
        bankWidth = 128,
        bankEntries = 16,
        bankMaskLen = 16,
        sharedEnable = true,
        sharedEntries = 16,
        sharedInputChannels = channelCount,
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
      top = base.top.copy(nCores = 1)
    )
  }

  private def init(c: SharedMemInputScheduler): Unit = {
    for (ch <- 0 until c.io.targetValid.length) {
      c.io.targetValid(ch).poke(false.B)
      c.io.targetPbank(ch).poke(0.U)

      c.io.inRead(ch).req.valid.poke(false.B)
      c.io.inRead(ch).req.bits.addr.poke(0.U)
      c.io.inRead(ch).resp.ready.poke(true.B)

      c.io.inWrite(ch).req.valid.poke(false.B)
      c.io.inWrite(ch).req.bits.addr.poke(0.U)
      c.io.inWrite(ch).req.bits.data.poke(0.U)
      c.io.inWrite(ch).req.bits.mask.foreach(_.poke(false.B))
      c.io.inWrite(ch).resp.ready.poke(true.B)
    }

    for (bank <- 0 until c.io.bankRead.length) {
      c.io.bankRead(bank).req.ready.poke(true.B)
      c.io.bankRead(bank).resp.valid.poke(false.B)
      c.io.bankRead(bank).resp.bits.data.poke(0.U)

      c.io.bankWrite(bank).req.ready.poke(true.B)
      c.io.bankWrite(bank).resp.valid.poke(false.B)
      c.io.bankWrite(bank).resp.bits.ok.poke(false.B)
    }
  }

  private def setTarget(c: SharedMemInputScheduler, ch: Int, bank: Int): Unit = {
    c.io.targetValid(ch).poke(true.B)
    c.io.targetPbank(ch).poke(bank.U)
  }

  private def driveRead(c: SharedMemInputScheduler, ch: Int, addr: Int): Unit = {
    c.io.inRead(ch).req.valid.poke(true.B)
    c.io.inRead(ch).req.bits.addr.poke(addr.U)
  }

  private def driveWrite(c: SharedMemInputScheduler, ch: Int, addr: Int, data: BigInt): Unit = {
    c.io.inWrite(ch).req.valid.poke(true.B)
    c.io.inWrite(ch).req.bits.addr.poke(addr.U)
    c.io.inWrite(ch).req.bits.data.poke(data.U)
    c.io.inWrite(ch).req.bits.mask.foreach(_.poke(true.B))
  }

  private def clearRequests(c: SharedMemInputScheduler): Unit = {
    for (ch <- 0 until c.io.inRead.length) {
      c.io.inRead(ch).req.valid.poke(false.B)
      c.io.inWrite(ch).req.valid.poke(false.B)
    }
  }

  private def expectNoBankIssue(c: SharedMemInputScheduler): Unit = {
    for (bank <- 0 until c.io.bankRead.length) {
      c.io.bankRead(bank).req.valid.expect(false.B)
      c.io.bankWrite(bank).req.valid.expect(false.B)
    }
  }

  it should "issue single-channel read and write requests and demux their responses" in {
    test(new SharedMemInputScheduler(schedulerConfig(), issueK)) { c =>
      init(c)

      setTarget(c, ch = 0, bank = 0)
      driveRead(c, ch = 0, addr = 3)

      c.io.inRead(0).req.ready.expect(true.B)
      c.io.bankRead(0).req.valid.expect(true.B)
      c.io.bankRead(0).req.bits.addr.expect(3.U)
      c.io.issuedCount.expect(1.U)

      c.clock.step()
      clearRequests(c)
      c.io.bankRead(0).resp.valid.poke(true.B)
      c.io.bankRead(0).resp.bits.data.poke(BigInt("123456789abcdef", 16).U)

      c.clock.step()
      c.io.bankRead(0).resp.valid.poke(false.B)
      c.io.inRead(0).resp.valid.expect(true.B)
      c.io.inRead(0).resp.bits.data.expect(BigInt("123456789abcdef", 16).U)
      for (ch <- 1 until channels) {
        c.io.inRead(ch).resp.valid.expect(false.B)
      }

      c.clock.step()

      setTarget(c, ch = 0, bank = 1)
      driveWrite(c, ch = 0, addr = 5, data = BigInt("feedfacecafebeef", 16))

      c.io.inWrite(0).req.ready.expect(true.B)
      c.io.bankWrite(1).req.valid.expect(true.B)
      c.io.bankWrite(1).req.bits.addr.expect(5.U)
      c.io.bankWrite(1).req.bits.data.expect(BigInt("feedfacecafebeef", 16).U)
      c.io.issuedCount.expect(1.U)

      c.clock.step()
      clearRequests(c)
      c.io.bankWrite(1).resp.valid.poke(true.B)
      c.io.bankWrite(1).resp.bits.ok.poke(true.B)

      c.clock.step()
      c.io.bankWrite(1).resp.valid.poke(false.B)
      c.io.inWrite(0).resp.valid.expect(true.B)
      c.io.inWrite(0).resp.bits.ok.expect(true.B)
      for (ch <- 1 until channels) {
        c.io.inWrite(ch).resp.valid.expect(false.B)
      }
    }
  }

  it should "issue different physical banks in parallel up to K ports" in {
    test(new SharedMemInputScheduler(schedulerConfig(), issueK)) { c =>
      init(c)
      setTarget(c, ch = 0, bank = 0)
      setTarget(c, ch = 1, bank = 1)
      driveWrite(c, ch = 0, addr = 0, data = 10)
      driveWrite(c, ch = 1, addr = 1, data = 11)

      c.io.inWrite(0).req.ready.expect(true.B)
      c.io.inWrite(1).req.ready.expect(true.B)
      c.io.bankWrite(0).req.valid.expect(true.B)
      c.io.bankWrite(1).req.valid.expect(true.B)
      c.io.issuedCount.expect(issueK.U)
      c.io.bankConflictStallCount.expect(0.U)
      c.io.noIssuePortStallCount.expect(0.U)
    }
  }

  it should "backpressure requests beyond K even when banks do not conflict" in {
    test(new SharedMemInputScheduler(schedulerConfig(), issueK)) { c =>
      init(c)
      setTarget(c, ch = 0, bank = 0)
      setTarget(c, ch = 1, bank = 1)
      setTarget(c, ch = 2, bank = 2)
      driveWrite(c, ch = 0, addr = 0, data = 10)
      driveWrite(c, ch = 1, addr = 1, data = 11)
      driveWrite(c, ch = 2, addr = 2, data = 12)

      c.io.inWrite(0).req.ready.expect(true.B)
      c.io.inWrite(1).req.ready.expect(true.B)
      c.io.inWrite(2).req.ready.expect(false.B)
      c.io.bankWrite(0).req.valid.expect(true.B)
      c.io.bankWrite(1).req.valid.expect(true.B)
      c.io.bankWrite(2).req.valid.expect(false.B)
      c.io.issuedCount.expect(issueK.U)
      c.io.noIssuePortStallCount.expect(1.U)
      c.io.bankConflictStallCount.expect(0.U)
    }
  }

  it should "not grant an input request until the selected bank is ready" in {
    test(new SharedMemInputScheduler(schedulerConfig(), issueK)) { c =>
      init(c)
      setTarget(c, ch = 0, bank = 0)
      driveWrite(c, ch = 0, addr = 0, data = 10)
      c.io.bankWrite(0).req.ready.poke(false.B)

      c.io.bankWrite(0).req.valid.expect(true.B)
      c.io.inWrite(0).req.ready.expect(false.B)
      c.io.issuedCount.expect(0.U)

      c.clock.step()
      c.io.bankWrite(0).req.ready.poke(true.B)

      c.io.bankWrite(0).req.valid.expect(true.B)
      c.io.inWrite(0).req.ready.expect(true.B)
      c.io.issuedCount.expect(1.U)
    }
  }

  it should "allow only one same-bank request in a cycle" in {
    test(new SharedMemInputScheduler(schedulerConfig(), issueK)) { c =>
      init(c)
      setTarget(c, ch = 0, bank = 0)
      setTarget(c, ch = 1, bank = 0)
      driveWrite(c, ch = 0, addr = 0, data = 10)
      driveWrite(c, ch = 1, addr = 1, data = 11)

      c.io.inWrite(0).req.ready.expect(true.B)
      c.io.inWrite(1).req.ready.expect(false.B)
      c.io.bankWrite(0).req.valid.expect(true.B)
      c.io.issuedCount.expect(1.U)
      c.io.bankConflictStallCount.expect(1.U)
      c.io.noIssuePortStallCount.expect(0.U)
    }
  }

  it should "treat read and write requests to the same bank as a single-port conflict" in {
    test(new SharedMemInputScheduler(schedulerConfig(), issueK)) { c =>
      init(c)
      setTarget(c, ch = 0, bank = 2)
      setTarget(c, ch = 1, bank = 2)
      driveRead(c, ch = 0, addr = 3)
      driveWrite(c, ch = 1, addr = 4, data = 22)

      c.io.inRead(0).req.ready.expect(true.B)
      c.io.inWrite(1).req.ready.expect(false.B)
      c.io.bankRead(2).req.valid.expect(true.B)
      c.io.bankWrite(2).req.valid.expect(false.B)
      c.io.issuedCount.expect(1.U)
      c.io.bankConflictStallCount.expect(1.U)
    }
  }

  it should "route simultaneous read responses back to their source channels" in {
    test(new SharedMemInputScheduler(schedulerConfig(), issueK)) { c =>
      init(c)
      setTarget(c, ch = 2, bank = 0)
      setTarget(c, ch = 3, bank = 1)
      driveRead(c, ch = 2, addr = 6)
      driveRead(c, ch = 3, addr = 7)

      c.io.inRead(2).req.ready.expect(true.B)
      c.io.inRead(3).req.ready.expect(true.B)
      c.io.bankRead(0).req.valid.expect(true.B)
      c.io.bankRead(1).req.valid.expect(true.B)
      c.clock.step()

      clearRequests(c)
      c.io.bankRead(0).resp.valid.poke(true.B)
      c.io.bankRead(0).resp.bits.data.poke(BigInt("abc", 16).U)
      c.io.bankRead(1).resp.valid.poke(true.B)
      c.io.bankRead(1).resp.bits.data.poke(BigInt("def", 16).U)

      c.clock.step()
      c.io.bankRead(0).resp.valid.poke(false.B)
      c.io.bankRead(1).resp.valid.poke(false.B)
      c.io.inRead(2).resp.valid.expect(true.B)
      c.io.inRead(2).resp.bits.data.expect(BigInt("abc", 16).U)
      c.io.inRead(3).resp.valid.expect(true.B)
      c.io.inRead(3).resp.bits.data.expect(BigInt("def", 16).U)
      c.io.inRead(0).resp.valid.expect(false.B)
      c.io.inRead(1).resp.valid.expect(false.B)
    }
  }

  it should "scan all channels correctly after the round-robin pointer wraps" in {
    val wrapChannels = 28
    val wrapBanks = 28

    test(new SharedMemInputScheduler(schedulerConfig(wrapChannels, wrapBanks), issuePortsK = 1)) { c =>
      init(c)
      setTarget(c, ch = 21, bank = 0)

      driveWrite(c, ch = 21, addr = 0, data = 21)
      c.io.inWrite(21).req.ready.expect(true.B)
      c.io.bankWrite(0).req.valid.expect(true.B)
      c.clock.step()

      clearRequests(c)
      c.clock.step()

      driveWrite(c, ch = 21, addr = 1, data = 22)
      c.io.inWrite(21).req.ready.expect(true.B)
      c.io.bankWrite(0).req.valid.expect(true.B)
      c.io.bankWrite(0).req.bits.addr.expect(1.U)
      c.io.issuedCount.expect(1.U)
    }
  }

  it should "hold target-invalid requests without issuing them to any bank" in {
    test(new SharedMemInputScheduler(schedulerConfig(), issueK)) { c =>
      init(c)
      driveRead(c, ch = 0, addr = 1)

      c.io.inRead(0).req.ready.expect(false.B)
      expectNoBankIssue(c)
      c.io.activeInputCount.expect(1.U)
      c.io.issuedCount.expect(0.U)
    }
  }
}
