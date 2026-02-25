#!/usr/bin/env bash
set -euo pipefail

if rg -n --glob '*.rs' --glob '!target/**' '#!\[allow\(|#\[allow\(' .; then
  echo "error: lint suppression via #[allow(...)] is forbidden; fix the lint instead."
  exit 1
fi

exit 0
