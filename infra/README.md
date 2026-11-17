# Published classroom results

Live deployment: https://red-smoke-04a79c703.4.azurestaticapps.net/

This Terraform configuration deploys a dedicated resource group and an Azure
Static Web App on the Free tier, with managed HTTPS and a real HTTP 404 page.
No training jobs, public model API, or raw corpus hosting is needed. Free-tier
quotas apply. No paid CDN, backend compute, or storage account is provisioned.

## Deploy or update

Requires Node.js 22+, npm 9+, Terraform 1.6+, Azure CLI, and `az login`. The
identity needs permission to create the resource group/Static Web App and read
its deployment token. The `Microsoft.Web` resource provider must be registered.
The deployment script downloads the pinned Static Web Apps CLI 2.1.0 through
`npm exec`; it does not install global packages.

Run from the repository root, choosing the intended subscription explicitly:

```bash
export AZURE_SUBSCRIPTION_ID="<your-subscription-id>"
bash scripts/deploy.sh
```

The script tests and builds an explicit publication allowlist, initializes and
validates Terraform, saves a plan, applies it, uploads the site, and verifies
every published file byte-for-byte over HTTPS, its MIME type, and the 404 page.
Terraform uses Azure CLI authentication. The deployer retrieves the Static Web
App deployment token into a short-lived environment variable, never a file,
CLI argument, browser asset, or terminal output. Keychain storage is disabled.
The published endpoint is intentionally public and read-only. This hosting
choice works without relaxing subscription policies that require private
storage accounts and disable storage shared keys.

The script applies the displayed plan without an interactive approval. To inspect
it separately, run `terraform -chdir=infra plan` with `TF_VAR_subscription_id` set.
Set `TF_VAR_location` before the first deployment to override `westeurope`.
Do not change the region of an existing deployment without reviewing replacement
actions.

Terraform state is local under `infra/`, contains the sensitive deployment token,
and is ignored by Git, along with saved plans, build outputs and credentials.
Preserve it securely between deployments; do not commit or publish it. Use a
secured remote backend before sharing deployment ownership or adding CI.
Commit `.terraform.lock.hcl` to preserve the provider selection/checksums.

Only generated website assets, selected lesson/report downloads, a projected
results JSON, the recorded microscope trace, hosting configuration, and a checksum manifest are deployed. Python source, checkpoints,
raw books, Azure credentials, Terraform state, and the working directory are
never uploaded. Unexpected files in `.site/` fail the build rather than being
silently published. Existing models/reports/decks are not regenerated or modified.
The microscope uses a committed CPU recording from `python3 build_microscope.py`.
Regenerating that recording requires the lab's Python/PyTorch dependencies, but
ordinary deployments do not. The website builder rejects a recording if its
original checkpoint or trace-producer source hashes have changed.

## Local checks

```bash
node --test scripts/site.test.mjs
node scripts/build-site.mjs
python3 -m http.server 8000 --bind 127.0.0.1 --directory .site
# Open http://localhost:8000 (Ctrl-C stops the local preview).
```

After deployment:

```bash
terraform -chdir=infra output -raw website_url
node scripts/verify-site.mjs "$(terraform -chdir=infra output -raw website_url)"
```

The page displays recorded inference, not interactive model generation.
HTTPS uses the native Azure Static Web Apps hostname; no custom domain, paid CDN,
analytics, cookies, or third-party browser dependencies are configured.
Browser-slide keyboard navigation and downloads are preserved.

## Remove the resources

Only when the website is no longer needed, set `TF_VAR_subscription_id` to the
same subscription and run `terraform -chdir=infra destroy`. Review the plan before
confirming; this removes this lab's resource group and Static Web App.
