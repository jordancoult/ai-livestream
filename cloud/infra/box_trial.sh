#!/bin/bash
# Rent a Vast offer and run the acceptance check before committing to it:
#   [PROFILE=joyai|ll|ll2|voice] cloud/infra/box_trial.sh OFFER_ID [pod file name, default .pod.trial]
# ACCEPT -> keeps it (pod file written). REJECT -> destroys it and blacklists its machine_id (with the reason).
set -u
cd "$(dirname "$0")/../.."
OFFER=$1; PF=cloud/infra/${2:-.pod.trial}; K=$(cat .secrets/vast_api_key); V=cloud/.venv/bin/vastai
ID=$(bash cloud/infra/vast.sh create "$OFFER" 2>/dev/null | awk '/^instance/{print $2}')
[ -z "$ID" ] && { echo "trial: create failed for offer $OFFER"; exit 2; }
echo "trial: instance $ID (offer $OFFER)"
for i in $(seq 1 60); do
  S=$($V --api-key "$K" --raw show instance "$ID" 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);p=(d.get('ports') or {}).get('22/tcp');print(d.get('actual_status'), d.get('public_ipaddr','').strip(), p[0]['HostPort'] if p else '', d.get('machine_id'), d.get('geolocation','').replace(' ','_'))" 2>/dev/null)
  echo "$S" | grep -q "^running .* [0-9]" && break; sleep 10
done
set -- $S; [ "${1:-}" = running ] || { echo "trial: never came up ($S): destroying"; bash cloud/infra/vast.sh destroy "$ID" >/dev/null 2>&1; exit 2; }
MID=$4; GEO=$5; echo "$2 $3" > "$PF"
for i in $(seq 1 15); do POD_FILE=$PF timeout 20 bash cloud/infra/pod.sh true 2>/dev/null && break; sleep 10; done
echo "trial: machine $MID ($GEO), ping from here: $(ping -c 4 -q "$2" | tail -1 | awk -F/ '{print $5}') ms"
OUT=$(bash cloud/infra/box_check.sh "$PF" "${PROFILE:-joyai}"); echo "$OUT" | sed 's/^VERDICT \([A-Z]*\).*/VERDICT \1/'
if echo "$OUT" | grep -q "^VERDICT ACCEPT"; then echo "trial: ACCEPTED instance $ID -> $PF"; exit 0; fi
WHY=$(echo "$OUT" | grep FAIL | sed 's/^ *FAIL *//' | tr -s ' ' | paste -sd ';' | cut -c1-160)
echo "$MID ${PROFILE:-joyai}  # $(date +%F) $GEO: box_check REJECT (${PROFILE:-joyai}): $WHY" >> cloud/infra/vast_blacklist.txt
bash cloud/infra/vast.sh destroy "$ID" >/dev/null 2>&1; rm -f "$PF"; echo "trial: REJECTED, destroyed, machine $MID blacklisted"; exit 1
