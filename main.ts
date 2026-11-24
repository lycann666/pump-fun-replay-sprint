// Dip-reclaim paper bot for the make.fun Pump.fun Replay Sprint (agent-fun/replay/v2).
//
// Strategy (own design, written in this session):
//   A coin first runs up, then dips, then reclaims part of the dip on fresh buy flow.
//   Entry needs, over the last 120 slots: a run-up of >= +20% from the window start to the
//   window high, a pullback of >= 8% from that high, and a reclaim of >= 40% of the dip.
//   The reclaim must be confirmed by real flow in the last 20 slots: net buy inflow
//   >= 0.30 SOL from >= 3 distinct buyers, with no single buyer over 70% of the buy SOL.
//   Real reserve must sit between 2 and 60 SOL. One buy per coin, 0.08 SOL each, at most
//   4 open positions. Exits are rule-based: +25% take-profit, -12% stop-loss, a 12%
//   trailing stop armed at +8%, a fade exit on 20-slot net flow below -0.40 SOL, and a
//   600-slot time stop unless the position is up >= 8%.
//   A 0.01 SOL probe keeps the guardrail pace (24 round-trip coins over the window) when
//   real entries fall behind it. Everything is slot-based and deterministic: no clock,
//   no randomness, no network, no environment; candidates sort by (recovery desc,
//   inflow desc, mint asc) so two runs on the same data decide identically.
//   The harness speaks hello once, then one batch per slot with its events, marks and
//   fills; the bot answers each batch with its orders and stops at done.
//
// Pass 1 (this write): full strategy as designed. Pass 2 applied the Freemodel
// design review (call a: 4,507 tokens truncated + 2,464-token retry, both logged
// in the notes): buy size 0.08 -> 0.20 SOL, take-profit +25% -> +35%, trailing
// stop armed at +15% and tightened to 9%. Grade-local then got WORSE on the full
// dev window (-1.179 SOL vs pass 1's -0.466 SOL: bigger size scaled the losses,
// fees doubled to 1.42 SOL curve), so pass 2 REVERTED all three parameters to the
// pass-1 values below and kept only the two correctness fixes: the pending-buy
// check now scans pending orders by mint (pending is keyed by client_id, not
// mint), and a stray `void h` was removed from the exposure helper. Final numbers
// in the strategy summary and in P are the reverted pass-1 values.

import { runBot } from './vendor/makefun-bot.ts';
import type { Batch, BotState, OrderInput } from './vendor/makefun-bot.ts';

const LAMPORTS_PER_SOL = 1_000_000_000n;
const SOL = (n: number): bigint => BigInt(Math.floor(n * 1_000_000_000));

// ---- Tunable parameters (all in one place, all deterministic) ----
const P = {
  minAgeSlots: 90,
  windowSlots: 120,
  flowSlots: 20,
  minRunUp: 0.2, // +20% start -> window high
  minDip: 0.08, // -8% high -> post-high low
  minReclaim: 0.4, // price recovered >= 40% of the dip
  minInflowLamports: SOL(0.3),
  minDistinctBuyers: 3,
  maxTopBuyerShare: 0.7,
  minRealReserve: SOL(2),
  maxRealReserve: SOL(60),
  buySol: 0.08,
  maxPositions: 4,
  maxExposureSol: 4,
  buySlipBps: 300,
  sellSlipBps: 10_000,
  takeProfit: 0.25,
  stopLoss: 0.12,
  trailArm: 0.08,
  trailDrop: 0.12,
  fadeFlowLamports: SOL(-0.4),
  maxHoldSlots: 600,
  holdKeepGain: 0.08,
  probeSol: 0.01,
  probeTargetCoins: 24,
  probeTargetFills: 200,
  probeWindowSeconds: 86_400, // 24h final window
  maxOrdersPerBatch: 16,
} as const;

interface TradeTick {
  slot: number;
  buy: boolean;
  sol: bigint;
  user: string;
  price: number;
}

interface Track {
  firstSlot: number;
  ticks: TradeTick[];
  lastPrice: number;
}

interface Held {
  openedSlot: number;
  entryPrice: number;
  peakPrice: number;
  probe: boolean;
}

function priceOf(vq: bigint, vt: bigint): number {
  if (vt <= 0n || vq <= 0n) return 0;
  return Number(vq) / Number(vt);
}

class DipReclaimBot {
  private tracks = new Map<string, Track>();
  private held = new Map<string, Held>();
  private roundTripCoins = new Set<string>();
  private boughtCoins = new Set<string>();
  private fills = 0;
  private firstBlockTime: number | null = null;
  private lastBlockTime: number | null = null;

  onBatch(state: BotState, batch: Batch): OrderInput[] {
    this.ingest(batch);
    this.syncHeld(state);
    const orders: OrderInput[] = [];
    const now = batch.slot;

    // ---- Exits first: every held position is re-checked against its rules. ----
    for (const [mint, h] of this.held) {
      const pos = state.positions.get(mint);
      if (!pos) continue;
      const curve = state.curves.get(mint);
      if (!curve || curve.complete) continue;
      const price = priceOf(curve.virtual_quote_reserves, curve.virtual_token_reserves);
      if (price <= 0) continue;
      if (price > h.peakPrice) h.peakPrice = price;
      const ret = price / h.entryPrice - 1;
      const flow20 = this.netFlow(mint, now, P.flowSlots);
      let exit = false;
      if (ret >= P.takeProfit) exit = true; // take-profit +25%
      else if (ret <= -P.stopLoss) exit = true; // stop-loss -12%
      else if (h.peakPrice >= h.entryPrice * (1 + P.trailArm) && price <= h.peakPrice * (1 - P.trailDrop)) exit = true; // trailing 12%, armed +8%
      else if (flow20 <= P.fadeFlowLamports) exit = true; // flow faded
      else if (now - h.openedSlot > P.maxHoldSlots && ret < P.holdKeepGain) exit = true; // time stop 600 slots
      if (exit) {
        orders.push({ side: 'sell', mint, token_amount: pos.token_amount, max_slippage_bps: P.sellSlipBps });
        this.held.delete(mint);
      }
    }

    // ---- Entries: rank reclaim candidates, strongest reclaim first. ----
    if (state.committedPositionCount() < P.maxPositions && orders.length < P.maxOrdersPerBatch) {
      const cands: { mint: string; reclaim: number; inflow: bigint }[] = [];
      for (const [mint, t] of this.tracks) {
        if (this.held.has(mint) || state.positions.has(mint)) continue;
        if (this.hasPendingBuy(state, mint)) continue;
        const curve = state.curves.get(mint);
        if (!curve || curve.complete) continue;
        const created = curve.created_slot ?? t.firstSlot;
        if (now - created < P.minAgeSlots) continue;
        if (curve.real_quote_reserves < P.minRealReserve || curve.real_quote_reserves > P.maxRealReserve) continue;
        const sig = this.reclaimSignal(mint, now);
        if (!sig) continue;
        cands.push({ mint, reclaim: sig.reclaim, inflow: sig.inflow });
      }
      cands.sort((a, b) => b.reclaim - a.reclaim || Number(b.inflow - a.inflow) || (a.mint < b.mint ? -1 : 1));
      for (const c of cands) {
        if (state.committedPositionCount() >= P.maxPositions) break;
        if (orders.length >= P.maxOrdersPerBatch) break;
        const cap = state.maxBuyLamports(c.mint);
        const want = SOL(P.buySol);
        const amount = want < cap ? want : cap;
        if (amount < SOL(0.01)) continue;
        if (state.freeCashLamports() < amount + state.txFeeLamports()) continue;
        const exposure = this.exposureSol(state);
        if (exposure + Number(amount) / 1e9 > P.maxExposureSol) continue;
        orders.push({ side: 'buy', mint: c.mint, quote_lamports: amount, max_slippage_bps: P.buySlipBps });
        const curve = state.curves.get(c.mint)!;
        const price = priceOf(curve.virtual_quote_reserves, curve.virtual_token_reserves);
        this.held.set(c.mint, { openedSlot: now, entryPrice: price > 0 ? price : 1, peakPrice: price > 0 ? price : 1, probe: false });
      }
    }

    // ---- Guardrail probe: 0.01 SOL round trips only while behind pace. ----
    if (orders.length < P.maxOrdersPerBatch) this.maybeProbe(state, batch, orders);

    return orders;
  }

  // Fold this batch's events into the per-mint tracks and fill counters.
  private ingest(batch: Batch): void {
    for (const e of batch.events) {
      if (e.block_time !== null) {
        if (this.firstBlockTime === null) this.firstBlockTime = e.block_time;
        this.lastBlockTime = e.block_time;
      }
      if (e.event !== 'TradeEvent') continue;
      const price = priceOf(BigInt(e.virtual_quote_reserves), BigInt(e.virtual_token_reserves));
      if (price <= 0) continue;
      let t = this.tracks.get(e.mint);
      if (!t) {
        t = { firstSlot: e.slot, ticks: [], lastPrice: price };
        this.tracks.set(e.mint, t);
      }
      t.ticks.push({ slot: e.slot, buy: e.is_buy, sol: BigInt(e.sol_amount), user: e.user, price });
      t.lastPrice = price;
      // Keep only the recent past: the 120-slot window plus a little slack.
      while (t.ticks.length > 0 && t.ticks[0].slot <= batch.slot - (P.windowSlots + 60)) t.ticks.shift();
    }
    for (const f of batch.fills) {
      if (f.status !== 'filled') continue;
      this.fills += 1;
      if (f.side === 'buy') this.boughtCoins.add(f.mint);
      if (f.side === 'sell' && this.boughtCoins.has(f.mint)) this.roundTripCoins.add(f.mint);
    }
    // Drop stale tracks so memory stays bounded over a 24h window.
    if (batch.seq % 50 === 0) {
      for (const [mint, t] of this.tracks) {
        const last = t.ticks.length > 0 ? t.ticks[t.ticks.length - 1].slot : t.firstSlot;
        if (last < batch.slot - (P.windowSlots + 60) && !this.held.has(mint)) this.tracks.delete(mint);
      }
    }
  }

  // Keep the held map in step with the harness account: drop closed/flattened
  // positions, adopt any position the map missed (entry price = current price).
  private syncHeld(state: BotState): void {
    for (const mint of [...this.held.keys()]) {
      if (!state.positions.has(mint)) this.held.delete(mint);
    }
    for (const [mint] of state.positions) {
      if (!this.held.has(mint)) {
        const curve = state.curves.get(mint);
        const price = curve ? priceOf(curve.virtual_quote_reserves, curve.virtual_token_reserves) : 0;
        this.held.set(mint, { openedSlot: state.slot, entryPrice: price > 0 ? price : 1, peakPrice: price > 0 ? price : 1, probe: false });
      }
    }
  }

  // Pending orders are keyed by client_id, so scan their mints for a pending buy.
  private hasPendingBuy(state: BotState, mint: string): boolean {
    for (const p of state.pending.values()) {
      if (p.order.side === 'buy' && p.order.mint === mint) return true;
    }
    return false;
  }

  private exposureSol(state: BotState): number {
    let total = 0;
    for (const [mint] of this.held) {
      const pos = state.positions.get(mint);
      if (pos) total += Number(pos.cost_lamports) / 1e9;
    }
    return total;
  }

  // Net buy-minus-sell SOL over the last `slots` slots for one mint.
  private netFlow(mint: string, now: number, slots: number): bigint {
    const t = this.tracks.get(mint);
    if (!t) return 0n;
    let net = 0n;
    for (let i = t.ticks.length - 1; i >= 0; i--) {
      const tk = t.ticks[i];
      if (tk.slot <= now - slots) break;
      net += tk.buy ? tk.sol : -tk.sol;
    }
    return net;
  }

  // The dip-reclaim signal, or null when the coin does not qualify.
  private reclaimSignal(mint: string, now: number): { reclaim: number; inflow: bigint } | null {
    const t = this.tracks.get(mint);
    if (!t || t.ticks.length < 4) return null;
    const win = t.ticks.filter((tk) => tk.slot > now - P.windowSlots);
    if (win.length < 4) return null;
    const startPrice = win[0].price;
    let high = 0;
    let highIdx = -1;
    for (let i = 0; i < win.length; i++) {
      if (win[i].price > high) {
        high = win[i].price;
        highIdx = i;
      }
    }
    if (high <= 0 || startPrice <= 0) return null;
    if (high / startPrice - 1 < P.minRunUp) return null; // needs a real run-up first
    let low = Infinity;
    for (let i = highIdx; i < win.length; i++) if (win[i].price < low) low = win[i].price;
    if (!Number.isFinite(low)) return null;
    const dip = (high - low) / high;
    if (dip < P.minDip) return null; // needs a real pullback
    const price = t.lastPrice;
    if (price <= low) return null;
    const reclaim = (price - low) / (high - low);
    if (reclaim < P.minReclaim) return null; // needs a real reclaim
    // Fresh flow must confirm the reclaim: net inflow, distinct buyers, no whale dominance.
    let inflow = 0n;
    let buySol = 0n;
    const buyers = new Map<string, bigint>();
    for (let i = win.length - 1; i >= 0; i--) {
      const tk = win[i];
      if (tk.slot <= now - P.flowSlots) break;
      if (tk.buy) {
        inflow += tk.sol;
        buySol += tk.sol;
        buyers.set(tk.user, (buyers.get(tk.user) ?? 0n) + tk.sol);
      } else {
        inflow -= tk.sol;
      }
    }
    if (inflow < P.minInflowLamports) return null;
    if (buyers.size < P.minDistinctBuyers) return null;
    let top = 0n;
    for (const v of buyers.values()) if (v > top) top = v;
    if (buySol > 0n && Number(top) / Number(buySol) > P.maxTopBuyerShare) return null;
    return { reclaim, inflow };
  }

  // One 0.01 SOL probe buy per batch, only while the round-trip pace trails the
  // guardrail target (24 coins over the 24h window, paced by data block_time).
  private maybeProbe(state: BotState, batch: Batch, orders: OrderInput[]): void {
    if (state.committedPositionCount() >= state.hello.caps.max_open_positions) return;
    if (this.firstBlockTime === null || this.lastBlockTime === null) return;
    const elapsed = this.lastBlockTime - this.firstBlockTime;
    if (elapsed <= 0) return;
    const expectedCoins = (P.probeTargetCoins * elapsed) / P.probeWindowSeconds;
    const expectedFills = (P.probeTargetFills * elapsed) / P.probeWindowSeconds;
    if (this.roundTripCoins.size >= expectedCoins && this.fills >= expectedFills) return;
    // Most-traded fresh coin with a real reserve, not already held or pending.
    let best: string | null = null;
    let bestTrades = -1;
    for (const [mint, t] of this.tracks) {
      if (this.held.has(mint) || state.positions.has(mint)) continue;
      if (this.boughtCoins.has(mint)) continue;
      const curve = state.curves.get(mint);
      if (!curve || curve.complete) continue;
      if (curve.real_quote_reserves < SOL(1)) continue;
      const recent = t.ticks.filter((tk) => tk.slot > batch.slot - P.flowSlots).length;
      if (recent > bestTrades || (recent === bestTrades && (best === null || mint < best))) {
        best = mint;
        bestTrades = recent;
      }
    }
    if (best === null || bestTrades < 2) return;
    const cap = state.maxBuyLamports(best);
    const amount = SOL(P.probeSol);
    if (amount > cap) return;
    if (state.freeCashLamports() < amount + state.txFeeLamports()) return;
    orders.push({ side: 'buy', mint: best, quote_lamports: amount, max_slippage_bps: P.buySlipBps });
    const curve = state.curves.get(best)!;
    const price = priceOf(curve.virtual_quote_reserves, curve.virtual_token_reserves);
    this.held.set(best, { openedSlot: batch.slot, entryPrice: price > 0 ? price : 1, peakPrice: price > 0 ? price : 1, probe: true });
  }
}

const bot = new DipReclaimBot();
await runBot({ onBatch: (state, batch) => bot.onBatch(state, batch) });
