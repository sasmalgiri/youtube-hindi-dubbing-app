"""Local Razorpay simulator (test mode only).

Imitates the two parts of Razorpay the app touches:
  * the REST API      POST /v1/orders, GET /v1/orders (Basic auth with the key id/secret)
  * the payment window GET /v1/checkout.js defines window.Razorpay with the same
                       options/handler/on('payment.failed')/open() contract

and sends signed webhooks (x-razorpay-signature = HMAC-SHA256 of the raw body
with the webhook secret, x-razorpay-event-id) to the app, like the real one.

Controls for tests (never in the real API):
  POST /sim/pay      {order_id, outcome: success|fail, webhook: normal|none|duplicate|delayed|before}
  POST /sim/refund   {payment_id}
  POST /sim/dispute  {payment_id, lost: bool}
  GET  /sim/state    everything the simulator holds

Run:  RAZORPAY_KEY_ID=rzp_test_sim RAZORPAY_KEY_SECRET=... RAZORPAY_WEBHOOK_SECRET=... \
      SIM_WEBHOOK_URL=http://127.0.0.1:8000/api/webhooks/razorpay \
      python -m uvicorn sim.razorpay_sim:app --port 8790
App:  RAZORPAY_API_BASE=http://127.0.0.1:8790/v1 (honoured only with rzp_test_ keys)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="Razorpay simulator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ORDERS: Dict[str, Dict[str, Any]] = {}
PAYMENTS: Dict[str, Dict[str, Any]] = {}
SENT: list = []
_lock = threading.Lock()


def _env(k: str) -> str:
    return os.environ.get(k, "").strip()


def _check_auth(request: Request) -> None:
    h = request.headers.get("authorization", "")
    try:
        user, pw = base64.b64decode(h.split(" ", 1)[1]).decode().split(":", 1)
    except Exception:
        raise HTTPException(401, {"error": {"code": "BAD_REQUEST_ERROR", "description": "Authentication failed"}})
    if user != _env("RAZORPAY_KEY_ID") or pw != _env("RAZORPAY_KEY_SECRET"):
        raise HTTPException(401, {"error": {"code": "BAD_REQUEST_ERROR", "description": "Authentication failed"}})
    if not user.startswith("rzp_test_"):
        raise HTTPException(400, {"error": {"description": "simulator accepts test keys only"}})


@app.post("/v1/orders")
async def create_order(request: Request):
    _check_auth(request)
    body = await request.json()
    amount = int(body.get("amount") or 0)
    if amount < 100:
        raise HTTPException(400, {"error": {"code": "BAD_REQUEST_ERROR", "description": "amount must be at least 100"}})
    oid = "order_sim" + secrets.token_hex(7)
    o = {"id": oid, "entity": "order", "amount": amount, "currency": body.get("currency", "INR"),
         "receipt": body.get("receipt"), "notes": body.get("notes") or {}, "status": "created",
         "created_at": int(time.time())}
    with _lock:
        ORDERS[oid] = o
    return o


@app.get("/v1/orders")
def list_orders(request: Request, count: int = 10):
    _check_auth(request)
    items = sorted(ORDERS.values(), key=lambda o: -o["created_at"])[:count]
    return {"entity": "collection", "count": len(items), "items": items}


def _sign(body: bytes) -> str:
    return hmac.new(_env("RAZORPAY_WEBHOOK_SECRET").encode(), body, hashlib.sha256).hexdigest()


def _post_webhook(event: str, payload: Dict[str, Any], event_id: Optional[str] = None) -> int:
    import requests
    url = _env("SIM_WEBHOOK_URL")
    if not url:
        return 0
    raw = json.dumps({"entity": "event", "event": event, "payload": payload, "created_at": int(time.time())}).encode()
    eid = event_id or "evt_sim" + secrets.token_hex(6)
    try:
        r = requests.post(url, data=raw, timeout=15, headers={"content-type": "application/json",
                                                               "x-razorpay-signature": _sign(raw),
                                                               "x-razorpay-event-id": eid})
        code = r.status_code
    except Exception:
        code = -1
    SENT.append({"event": event, "event_id": eid, "status": code})
    return code


class PayBody(BaseModel):
    order_id: str
    outcome: str = "success"         # success | fail
    webhook: str = "normal"          # normal | none | duplicate | delayed | before


@app.post("/sim/pay")
def pay(b: PayBody):
    with _lock:
        o = ORDERS.get(b.order_id)
        if not o:
            raise HTTPException(404, "no such order")
        if b.outcome != "success":
            return {"error": {"code": "BAD_REQUEST_ERROR", "description": "Card declined (simulated)",
                              "reason": "payment_failed"}}
        pid = "pay_sim" + secrets.token_hex(7)
        o["status"] = "paid"
        PAYMENTS[pid] = {"id": pid, "order_id": o["id"], "amount": o["amount"], "status": "captured"}
    sig = hmac.new(_env("RAZORPAY_KEY_SECRET").encode(), f"{o['id']}|{pid}".encode(), hashlib.sha256).hexdigest()
    payload = {"payment": {"entity": {"id": pid, "order_id": o["id"], "amount": o["amount"], "status": "captured"}},
               "order": {"entity": {"id": o["id"], "amount": o["amount"], "status": "paid"}}}

    def send():
        if b.webhook == "delayed":
            time.sleep(3)
        if b.webhook != "none":
            _post_webhook("payment.captured", payload)
            _post_webhook("order.paid", payload)
        if b.webhook == "duplicate":
            ev = SENT[-1]["event_id"] if SENT else None
            _post_webhook("order.paid", payload, event_id=ev)
    if b.webhook == "before":
        send()                       # webhook lands before the browser confirms
    else:
        threading.Thread(target=send, daemon=True).start()
    return {"razorpay_order_id": o["id"], "razorpay_payment_id": pid, "razorpay_signature": sig}


class PaymentRef(BaseModel):
    payment_id: str
    lost: bool = False


@app.post("/sim/refund")
def refund(b: PaymentRef):
    p = PAYMENTS.get(b.payment_id)
    if not p:
        raise HTTPException(404, "no such payment")
    rid = "rfnd_sim" + secrets.token_hex(6)
    p["status"] = "refunded"
    code = _post_webhook("refund.processed", {"refund": {"entity": {"id": rid, "payment_id": p["id"], "amount": p["amount"]}},
                                              "payment": {"entity": p}})
    return {"refund_id": rid, "webhook_status": code}


@app.post("/sim/dispute")
def dispute(b: PaymentRef):
    p = PAYMENTS.get(b.payment_id)
    if not p:
        raise HTTPException(404, "no such payment")
    did = "disp_sim" + secrets.token_hex(6)
    ev = "payment.dispute.lost" if b.lost else "payment.dispute.created"
    code = _post_webhook(ev, {"dispute": {"entity": {"id": did, "payment_id": p["id"], "amount": p["amount"]}},
                              "payment": {"entity": p}})
    return {"dispute_id": did, "webhook_status": code}


@app.get("/sim/state")
def state():
    return {"orders": list(ORDERS.values()), "payments": list(PAYMENTS.values()), "webhooks": SENT}


CHECKOUT_JS = r"""
// Razorpay checkout simulator: same constructor/handler contract as checkout.razorpay.com/v1/checkout.js
(function () {
  var SIM = %SIM%;
  function Razorpay(opts) { this.o = opts; this.h = {}; }
  Razorpay.prototype.on = function (ev, f) { this.h[ev] = f; };
  Razorpay.prototype.open = function () {
    var self = this, o = this.o;
    var wrap = document.createElement('div');
    wrap.setAttribute('role', 'dialog'); wrap.setAttribute('aria-label', 'Simulated payment');
    wrap.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:99999';
    wrap.innerHTML = '<div style="background:#fff;color:#111;padding:20px;border-radius:10px;min-width:300px;font:14px sans-serif">' +
      '<b>SIMULATED PAYMENT</b><p>' + (o.description || '') + '<br>' + (o.amount / 100).toFixed(2) + ' ' + o.currency + '</p>' +
      '<button data-x="pay">Pay (simulated)</button> <button data-x="fail">Decline</button> <button data-x="close">Close</button></div>';
    document.body.appendChild(wrap);
    function done() { wrap.remove(); }
    wrap.querySelector('[data-x=close]').onclick = function () { done(); o.modal && o.modal.ondismiss && o.modal.ondismiss(); };
    function go(outcome) {
      fetch(SIM + '/sim/pay', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ order_id: o.order_id, outcome: outcome, webhook: (window.__rzpSimWebhook || 'normal') }) })
        .then(function (r) { return r.json(); })
        .then(function (r) {
          done();
          if (r.error) { self.h['payment.failed'] && self.h['payment.failed']({ error: r.error }); return; }
          o.handler && o.handler(r);
        });
    }
    wrap.querySelector('[data-x=pay]').onclick = function () { go('success'); };
    wrap.querySelector('[data-x=fail]').onclick = function () { go('fail'); };
  };
  window.Razorpay = Razorpay;
})();
"""


@app.get("/v1/checkout.js")
def checkout_js(request: Request):
    base = str(request.base_url).rstrip("/")
    return Response(CHECKOUT_JS.replace("%SIM%", json.dumps(base)), media_type="application/javascript")
