"""Local operator reconciliation only; never exposed as a model/API tool."""
import argparse
import json
from pathlib import Path

from .vision_jobs import Ledger, error


def reconcile(ledger, identity, receipt):
    with ledger.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM jobs WHERE id=?',(identity,)).fetchone()
        if (row is None or row['state'] != 'unknown' or receipt.get('job_id') != identity
                or receipt.get('backend_request_id') != row['backend_id']
                or receipt.get('termination_confirmed') is not True
                or len(str(receipt.get('evidence','')).strip()) < 32):
            raise ValueError('Unknown owned job and concrete verified backend termination receipt required')
        db.execute('CREATE TABLE IF NOT EXISTS reconciliation(id TEXT, recorded REAL, receipt TEXT)')
        db.execute('INSERT INTO reconciliation VALUES(?,?,?)',(identity,ledger.clock(),json.dumps(receipt)))
        result = {'status_code':422,'headers':{},'data':error('operator_reconciled',
            'Operator verified termination. This completed job will not be dispatched again.',identity)}
        db.execute("UPDATE jobs SET state='failed',updated=?,result=? WHERE id=?",
                   (ledger.clock(),json.dumps(result),identity))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--verified-termination-receipt',type=Path)
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error('Existing ledger required; do not create an empty replacement')
    ledger = Ledger(args.db)
    if args.verified_termination_receipt:
        reconcile(ledger,args.job_id,json.loads(args.verified_termination_receipt.read_text(encoding='utf-8')))
    row = ledger.read(args.job_id)
    if not row:
        parser.error('Unknown job ID')
    print(json.dumps(ledger.status(args.job_id,row['owner']),indent=2))


if __name__ == '__main__':
    main()
