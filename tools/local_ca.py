"""A private, name-constrained certificate authority for the phone pages (tools/phone_view.py, tools/phone_link).

Safari on iOS accepts a self-signed page certificate after a warning, but its WebSockets don't reliably honour that
exception (they hang in "connecting"; seen 2026-11-26 on the iPhone 13 Pro: fetch 200, WebSocket never connects).
With this CA installed and trusted on the phone once, every laptop page and WebSocket is trusted: no warning, no hang.

Safety: the CA is NAME-CONSTRAINED (critical): it can only vouch for private IPv4 addresses (10/8, 172.16/12,
192.168/16) and the name "localhost". A public website (any other DNS name, any public IP) is outside its permitted
subtrees, so even a leaked key can't impersonate real sites. pathlen 0: it can't create sub-CAs. The key stays on
this laptop in .secrets/local_ca/ (600, gitignored). Remove from the phone: Settings > General > VPN & Device
Management > the profile > Remove.

  cloud/facefix/.venv/bin/python tools/local_ca.py            # create the CA (once) and print where it is
  from local_ca import server_cert; crt, key = server_cert(["192.168.8.186", ...], out_dir)
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CA_DIR = REPO / ".secrets" / "local_ca"
CA_KEY, CA_CRT = CA_DIR / "ca.key", CA_DIR / "ca.pem"
CA_DER = CA_DIR / "ai-livestream-local-ca.crt"     # the file the phone installs
CONSTRAINTS = ("critical,permitted;IP:192.168.0.0/255.255.0.0,permitted;IP:10.0.0.0/255.0.0.0,"
               "permitted;IP:172.16.0.0/255.240.0.0,permitted;DNS:localhost")


def ensure_ca():
    if CA_KEY.exists() and CA_CRT.exists():
        return CA_CRT
    CA_DIR.mkdir(parents=True, exist_ok=True)
    CA_DIR.chmod(0o700)
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:3072", "-nodes", "-days", "3650", "-sha256",
                    "-keyout", str(CA_KEY), "-out", str(CA_CRT), "-subj", "/CN=ai-livestream local CA (LAN only)",
                    "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
                    "-addext", "keyUsage=critical,keyCertSign,cRLSign",
                    "-addext", f"nameConstraints={CONSTRAINTS}"], check=True, capture_output=True)
    CA_KEY.chmod(0o600)
    subprocess.run(["openssl", "x509", "-in", str(CA_CRT), "-outform", "DER", "-out", str(CA_DER)], check=True)
    return CA_CRT


def server_cert(ips, out_dir):
    """A leaf certificate for these IPs (+ localhost) signed by the CA; iOS rules: <= 825 days, serverAuth, SAN."""
    ensure_ca()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crt, key, tag = out_dir / "ca_cert.pem", out_dir / "ca_key.pem", out_dir / "ca_ips.txt"
    want = ",".join(sorted(ips))
    if crt.exists() and key.exists() and tag.exists() and tag.read_text() == want:
        return crt, key
    csr = out_dir / "req.csr"
    ext = out_dir / "ext.cnf"
    ext.write_text("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
                   "extendedKeyUsage=serverAuth\nsubjectAltName=" +
                   ",".join([f"IP:{i}" for i in ips] + ["DNS:localhost"]) + "\n")
    subprocess.run(["openssl", "req", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(csr),
                    "-subj", "/CN=ai-livestream phone link"], check=True, capture_output=True)
    subprocess.run(["openssl", "x509", "-req", "-in", str(csr), "-CA", str(CA_CRT), "-CAkey", str(CA_KEY),
                    "-CAcreateserial", "-CAserial", str(CA_DIR / "ca.srl"), "-days", "800", "-sha256",
                    "-extfile", str(ext), "-out", str(crt)], check=True, capture_output=True)
    key.chmod(0o600)
    csr.unlink(missing_ok=True)
    tag.write_text(want)
    return crt, key


if __name__ == "__main__":
    ensure_ca()
    print(f"CA certificate (install on the phone): {CA_DER}")
    print(subprocess.run(["openssl", "x509", "-in", str(CA_CRT), "-noout", "-subject", "-enddate", "-ext",
                          "nameConstraints,basicConstraints"], capture_output=True, text=True).stdout)
    sys.exit(0)
