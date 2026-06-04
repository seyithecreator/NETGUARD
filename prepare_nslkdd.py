"""
prepare_nslkdd.py
─────────────────────────────────────────────────────────────────────────────
Converts raw NSL-KDD .TXT files (KDDTrain+.txt / KDDTest+.txt) into a
single clean CSV with:
  • proper column headers
  • `attack_type`  column  (original string label e.g. 'neptune')
  • `label`        column  (integer 0-4 mapping used by the IDS)

Usage:
  python prepare_nslkdd.py                          # uses default paths below
  python prepare_nslkdd.py --train KDDTrain+.txt --test KDDTest+.txt --out nslkdd.csv

Output:
  nslkdd.csv  →  drop this into  netguard/data/nslkdd.csv
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import argparse, sys, os

# ── Column names (standard NSL-KDD 41 features + attack + difficulty) ────────
COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'attack_type','difficulty'          # last two columns in the raw file
]

# ── Attack → integer label mapping (matches your IDS) ────────────────────────
#   0 = Normal
#   1 = DoS
#   2 = Probe
#   3 = R2L (Remote-to-Local)
#   4 = U2R (User-to-Root)

ATTACK_MAP = {
    # ── Normal ──────────────────────────────────────────────────
    'normal':             0,

    # ── DoS ─────────────────────────────────────────────────────
    'back':               1, 'land':               1,
    'neptune':            1, 'pod':                1,
    'smurf':              1, 'teardrop':           1,
    'apache2':            1, 'udpstorm':           1,
    'processtable':       1, 'worm':               1,
    'mailbomb':           1,

    # ── Probe ────────────────────────────────────────────────────
    'ipsweep':            2, 'mscan':              2,
    'nmap':               2, 'portsweep':          2,
    'saint':              2, 'satan':              2,

    # ── R2L ─────────────────────────────────────────────────────
    'ftp_write':          3, 'guess_passwd':       3,
    'imap':               3, 'multihop':           3,
    'named':              3, 'phf':                3,
    'sendmail':           3, 'snmpgetattack':      3,
    'snmpguess':          3, 'spy':                3,
    'warezclient':        3, 'warezmaster':        3,
    'xlock':              3, 'xsnoop':             3,
    'httptunnel':         3,

    # ── U2R ─────────────────────────────────────────────────────
    'buffer_overflow':    4, 'loadmodule':         4,
    'perl':               4, 'ps':                 4,
    'rootkit':            4, 'sqlattack':          4,
    'xterm':              4,
}


def load_txt(path: str) -> pd.DataFrame:
    """Load a raw NSL-KDD .TXT file and attach column headers."""
    print(f"  Loading {path} …")
    df = pd.read_csv(path, header=None, names=COLUMNS, low_memory=False)
    print(f"  → {len(df):,} rows loaded")
    return df


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer `label` column from `attack_type`."""
    df['attack_type'] = df['attack_type'].str.strip().str.lower()

    unknown = set(df['attack_type'].unique()) - set(ATTACK_MAP.keys())
    if unknown:
        print(f"  ⚠  Unknown attack types found (will map to -1): {unknown}")

    df['label'] = df['attack_type'].map(ATTACK_MAP).fillna(-1).astype(int)

    # Summary
    mapping_friendly = {0:'Normal', 1:'DoS', 2:'Probe', 3:'R2L', 4:'U2R'}
    dist = df['label'].map(mapping_friendly).value_counts()
    print("  Label distribution:")
    for name, count in dist.items():
        print(f"    {name:<10} {count:>7,}")
    unmapped = (df['label'] == -1).sum()
    if unmapped:
        print(f"    {'UNMAPPED':<10} {unmapped:>7,}  ← check ATTACK_MAP")
    return df


def prepare(train_path, test_path, out_path):
    frames = []

    if train_path and os.path.exists(train_path):
        print(f"\n[Train] {train_path}")
        df_tr = load_txt(train_path)
        df_tr = add_label(df_tr)
        df_tr['partition'] = 'train'
        frames.append(df_tr)
    elif train_path:
        print(f"[WARN] Train file not found: {train_path}")

    if test_path and os.path.exists(test_path):
        print(f"\n[Test]  {test_path}")
        df_te = load_txt(test_path)
        df_te = add_label(df_te)
        df_te['partition'] = 'test'
        frames.append(df_te)
    elif test_path:
        print(f"[WARN] Test file not found: {test_path}")

    if not frames:
        print("ERROR: No input files found. Check your paths and try again.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Drop the 'difficulty' column — not used by the IDS
    combined.drop(columns=['difficulty'], errors='ignore', inplace=True)

    combined.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(combined):,} total rows → {out_path}")
    print(f"  Columns: {list(combined.columns)}")
    print(f"\nNext step: copy {out_path} into  netguard/data/nslkdd.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare NSL-KDD dataset for NetGuard IDS')
    parser.add_argument('--train', default='KDDTrain+.txt', help='Path to KDDTrain+.txt')
    parser.add_argument('--test',  default='KDDTest+.txt',  help='Path to KDDTest+.txt')
    parser.add_argument('--out',   default='nslkdd.csv',    help='Output CSV path')
    args = parser.parse_args()

    print("=" * 60)
    print("  NetGuard IDS — NSL-KDD Dataset Preparation")
    print("=" * 60)
    prepare(args.train, args.test, args.out)
