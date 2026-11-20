"""Prepaid time: catalog, quotes and the immutable ledger (tracker M01, M02, M04, P06).

Units, everywhere: **time in integer seconds**, **money in integer minor units**
of its own currency (paise for INR). Hours and currency never share a field.

    catalog.py  packs, price versions, rates (one documented rounding rule)
    quote.py    a quote bound to everything that changes its price
    ledger.py   grant / reserve / settle / release / refund / adjust / chargeback,
                unique business keys, single-transaction reservation

Razorpay (M03), refunds to the original payment method (M05) and accounts (S01)
plug into these; nothing here talks to a payment provider.
"""
