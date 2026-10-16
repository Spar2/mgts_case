"""Audit the local hardening handoff before publication.

This verifies the named artifacts, source fingerprints and frozen results.
It is not an independent scientific ground truth or a production certificate.
"""
import hashlib
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fiber.classify.physical import FEATURE_NAMES, load_physical_model
from fiber.pipeline import PipelineConfig


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    native=read(ROOT/'artifacts/hardening/real-data-verification.json')
    physical=read(ROOT/'artifacts/hardening/physical/report.json')
    annotation=read(ROOT/'artifacts/hardening/annotation/manifest.json')
    model=ROOT/'artifacts/hardening/physical/physical_model.json'
    decoded=load_physical_model(model)
    proposals=[json.loads(line) for line in (ROOT/'artifacts/hardening/annotation/proposals.jsonl').read_text().splitlines()]
    blind=[json.loads(line) for line in (ROOT/'artifacts/hardening/annotation/blind_review.jsonl').read_text().splitlines()]
    tests=ET.parse(ROOT/'artifacts/hardening/tests-verified.xml').getroot().find('testsuite')
    contracts=ET.parse(ROOT/'artifacts/hardening/final-contract-tests.xml').getroot().find('testsuite')
    checks={
        'six_records_verified':len(native['records'])==6 and native['all_checks_pass'],
        'native_manifest_current':native['manifest']==PipelineConfig().manifest(),
        'no_zone_across_time_gaps':all(r['zones_across_gaps']==0 for r in native['records']),
        'source_fingerprints_current':physical['feature_code_sha256']==sha(ROOT/'fiber/classify/physical.py') and physical['script_sha256']==sha(ROOT/'scripts/benchmark_physical.py'),
        'model_fingerprint':physical['model_sha256']==sha(model),
        'feature_schema':physical['features']==list(FEATURE_NAMES),
        'disjoint_seed_groups':not(set(physical['train_seeds'])&set(physical['validation_seeds'])) and not(set(physical['test_seeds'])&(set(physical['train_seeds'])|set(physical['validation_seeds']))),
        'model_provenance':decoded['expert_confirmed'] is False and decoded['serve']=='research_offline',
        'queue_complete':len(proposals)==len(blind)==annotation['n_items']==268,
        'queue_has_no_fabricated_expert_labels':all(row['expert_label'] is None and row['expert_confirmed'] is False for row in proposals+blind),
        'blind_proposals_hidden':all('proposed_shape' not in row and 'origin' not in row and row['id'].startswith('review-') for row in blind),
        'unique_blind_ids':len({row['id'] for row in blind})==len(blind),
        'all_tests_no_failures':int(tests.attrib['failures'])==0 and int(tests.attrib['errors'])==0,
        'final_contracts_no_failures':int(contracts.attrib['failures'])==0 and int(contracts.attrib['errors'])==0,
        'illustrations_exist':all((ROOT/'docs/images/hardening'/name).exists() for name in ('temporal_order.png','physical_confusion.png')),
        'handoff_exists':all((ROOT/'docs'/name).exists() for name in ('ML_HARDENING.md','ANNOTATION_PROTOCOL.md','HARDENING_PLAN.md')),
    }
    source={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
            for folder in ('fiber','scripts','tests') for path in (ROOT/folder).rglob('*.py') if '__pycache__' not in path.parts}
    report={'schema':'hardening_release_audit_v1','checks':checks,'all_checks_pass':all(checks.values()),
            'test_summary':tests.attrib,'final_contract_test_summary':contracts.attrib,
            'native_zones':[{k:r[k] for k in ('record','before_zones','after_zones','zones_across_gaps')} for r in native['records']],
            'model_sha256':physical['model_sha256'],'model_status':'research_offline',
            'minimum_class_test_recall':min(physical['test']['per_class_recall'].values()),
            'annotation_items':len(blind),'expert_confirmed':False,
            'source_hash_convention':'utf8_text_universal_newlines','source_sha256':source}
    dest=ROOT/'docs/evidence/hardening_release_audit.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'all_checks_pass':report['all_checks_pass'],'checks':checks}))
    if not report['all_checks_pass']: raise SystemExit(1)


if __name__=='__main__': main()
