#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trap 'rm -f "$ROOT/infra/website.tfplan"; unset SWA_CLI_DEPLOYMENT_TOKEN' EXIT
cd "$ROOT"
if [[ -z "${AZURE_SUBSCRIPTION_ID:-}" ]]; then
  echo "Set AZURE_SUBSCRIPTION_ID explicitly to the intended Azure subscription." >&2
  exit 1
fi
export TF_VAR_subscription_id="$AZURE_SUBSCRIPTION_ID"

az account show --subscription "$AZURE_SUBSCRIPTION_ID" --query '{name:name,id:id}' -o table
node --test scripts/site.test.mjs
node scripts/build-site.mjs
terraform -chdir=infra init -input=false
terraform -chdir=infra validate
terraform -chdir=infra plan -input=false -out=website.tfplan
terraform -chdir=infra apply -input=false website.tfplan
APP="$(terraform -chdir=infra output -raw static_web_app_name)"
GROUP="$(terraform -chdir=infra output -raw resource_group_name)"
URL="$(terraform -chdir=infra output -raw website_url)"

export SWA_CLI_DEPLOYMENT_TOKEN
SWA_CLI_DEPLOYMENT_TOKEN="$(az staticwebapp secrets list \
  --subscription "$AZURE_SUBSCRIPTION_ID" --name "$APP" --resource-group "$GROUP" \
  --query properties.apiKey --output tsv)"
npm exec --yes --package=@azure/static-web-apps-cli@2.1.0 -- \
  swa deploy "$ROOT/.site" --swa-config-location "$ROOT/web" --env production --no-use-keychain
unset SWA_CLI_DEPLOYMENT_TOKEN

for attempt in 1 2 3 4 5; do
  if node scripts/verify-site.mjs "$URL"; then
    printf '\nPublished website: %s\nResource group: %s\n' "$URL" "$GROUP"
    exit 0
  fi
  if [[ "$attempt" == 5 ]]; then
    echo "Deployment completed, but public verification failed." >&2
    exit 1
  fi
  echo "Public endpoint not ready; retrying verification in 10 seconds ($attempt/5)." >&2
  sleep 10
done
