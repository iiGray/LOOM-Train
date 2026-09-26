#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash ray_cluster.sh <head|worker> [OPTIONS] [-- RAY_START_OPTIONS...]

Start Ray on the current machine. Run once per node before submitting training.

  head                  Create a single-node or multi-node cluster.
  worker                Join an existing head (requires --master-ip).
  --master-ip IP        Head's local IP for head; head's reachable IP for worker.
                        Omit for head to let Ray detect the local IP.
  --ray-port PORT       Head port, shared with ray_launch.py (default: 6381).
  -h, --help            Show this help.
  --                    Forward remaining options directly to ray start.

Examples:
  bash ray_cluster.sh head
  bash ray_cluster.sh head --master-ip 10.0.0.10
  bash ray_cluster.sh worker --master-ip 10.0.0.10

CPU/GPU resources are detected by Ray unless overridden after --.
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

require_value() {
    [[ -n ${2:-} && ${2:-} != -* ]] || die "$1 requires a value"
}

case "${1:-}" in
    head|worker) role=$1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

master_ip=""
ray_port=6381
while (( $# )); do
    case "$1" in
        --master-ip)
            require_value "$@"
            master_ip=$2
            shift 2
            ;;
        --ray-port)
            require_value "$@"
            ray_port=$2
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) die "Unknown option: $1 (use -- before extra ray start options)" ;;
    esac
done

[[ $ray_port =~ ^[0-9]{1,5}$ ]] && (( 10#$ray_port >= 1 && 10#$ray_port <= 65535 )) \
    || die "--ray-port must be between 1 and 65535"

start_args=(start)
if [[ $role == head ]]; then
    start_args+=(--head --port "$ray_port")
    if [[ -n $master_ip ]]; then
        start_args+=(--node-ip-address "$master_ip")
    fi
else
    [[ -n $master_ip ]] || die "worker requires --master-ip pointing to the head"
    start_args+=(--address "$master_ip:$ray_port")
fi

command -v ray >/dev/null 2>&1 || die "Install Ray in the active environment: python -m pip install ray"
exec ray "${start_args[@]}" "$@"
