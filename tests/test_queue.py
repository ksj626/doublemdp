import json
from doublemdp.sweep import connect,enqueue,claim,status


def test_queue_claim_is_exclusive_and_preserves_pending(tmp_path):
    db=connect(tmp_path/"queue.db")
    for i in range(3):
        enqueue(db,str(i),{"output":str(tmp_path/str(i))},priority=3-i)
    assert claim(db,"0")["id"]=="2"
    second=connect(tmp_path/"queue.db")
    assert claim(second,"1")["id"]=="1"
    records=status(db)
    assert [r["status"] for r in records]==["running","running","pending"]


def test_enqueue_is_idempotent(tmp_path):
    db=connect(tmp_path/"queue.db")
    enqueue(db,"same",{"output":str(tmp_path/"run")})
    enqueue(db,"same",{"output":str(tmp_path/"run")})
    assert len(status(db))==1
