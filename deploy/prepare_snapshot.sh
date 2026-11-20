#!/usr/bin/env bash
# Cleans the VM before a snapshot is taken in the immers.cloud panel.
# cloud-init runs again on every VM made from the snapshot: new SSH host keys
# (each VM must be added to gpu-harbor's known_hosts on its own), keypair injection.
# The VM powers off at the end; take the snapshot of the stopped VM, then delete it.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }
apt-get clean
rm -rf /tmp/* /var/tmp/* /home/trainer/.cache/uv
journalctl --rotate && journalctl --vacuum-time=1s || true
cloud-init clean --logs
rm -f /root/.bash_history /home/*/.bash_history
sync
echo "Powering off. Snapshot it in the panel, wait for the image to become active, then delete the VM."
poweroff
