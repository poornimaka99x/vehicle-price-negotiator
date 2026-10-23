#!/usr/bin/env bash
# One-off Azure setup for the MLOps pipeline. Safe to re-run: existing resources are kept.
# Needs: az (logged in to the target subscription), gh (logged in, run from this repo).
set -euo pipefail

LOCATION=${LOCATION:-centralindia}          # the Students subscription only allows a few regions
RG=${RG:-rg-vehicle-price}
SUB=$(az account show --query id -o tsv)
SUFFIX=${SUB:0:6}
ACR=${ACR:-acrvehicleprice$SUFFIX}
AML=${AML:-mlw-vehicle-price}
KV=${KV:-kvvehicleprice$SUFFIX}
ENV=${ENV:-cae-vehicle-price}
APP=${APP:-vehicle-price}
GH_APP=${GH_APP:-gh-vehicle-price-negotiator}
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
BUDGET=${BUDGET:-10}                         # monthly, in the subscription's currency
ALERT_EMAIL=${ALERT_EMAIL:-$(az account show --query user.name -o tsv)}

for ns in Microsoft.App Microsoft.ContainerRegistry Microsoft.MachineLearningServices \
          Microsoft.Storage Microsoft.KeyVault Microsoft.Insights Microsoft.OperationalInsights; do
  az provider register -n "$ns" --wait -o none
done
az extension add -n ml -y -o none   # containerapp commands are in core az; no extension needed

az group create -n "$RG" -l "$LOCATION" -o none
# Admin user: express Container Apps environments cannot pull with a managed identity.
az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled true -o none
ACR_ID=$(az acr show -n "$ACR" --query id -o tsv)

# New key vaults default to RBAC, but workspace creation writes access policies, so bring our own vault.
az keyvault show -n "$KV" -o none 2>/dev/null ||
  az keyvault create -n "$KV" -g "$RG" -l "$LOCATION" -o none
az keyvault update -n "$KV" --enable-rbac-authorization false -o none

# Workspace = model registry. It also creates its own storage and app insights.
az ml workspace show -n "$AML" -g "$RG" -o none 2>/dev/null ||
  az ml workspace create -n "$AML" -g "$RG" -l "$LOCATION" --container-registry "$ACR_ID" \
    --key-vault "$(az keyvault show -n "$KV" --query id -o tsv)" -o none

# No Log Analytics: streaming logs (az containerapp logs show) are enough here and cost nothing.
az containerapp env show -n "$ENV" -g "$RG" -o none 2>/dev/null ||
  az containerapp env create -n "$ENV" -g "$RG" -l "$LOCATION" --logs-destination none -o none

# Placeholder image; the pipeline deploys the real one. Scales to zero when idle.
az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null ||
  az containerapp create -n "$APP" -g "$RG" --environment "$ENV" \
    --image mcr.microsoft.com/k8se/quickstart:latest \
    --ingress external --target-port 8501 --min-replicas 0 --max-replicas 1 \
    --cpu 1 --memory 2Gi -o none
# Stored as a Container App secret; GitHub never sees it.
az containerapp registry set -n "$APP" -g "$RG" --server "$ACR.azurecr.io" \
  --username "$ACR" --password "$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)" -o none

# GitHub Actions -> Azure with OIDC: a federated credential for pushes to main, no client secret.
APP_ID=$(az ad app list --display-name "$GH_APP" --query "[0].appId" -o tsv)
if [ -z "$APP_ID" ]; then
  APP_ID=$(az ad app create --display-name "$GH_APP" --query appId -o tsv)
  az ad sp create --id "$APP_ID" -o none
  az ad app federated-credential create --id "$APP_ID" --parameters "{
    \"name\": \"main\", \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"repo:$REPO:ref:refs/heads/main\", \"audiences\": [\"api://AzureADTokenExchange\"]}" -o none
fi
# Scoped to this resource group only. Contributor does not include pushing images, hence AcrPush.
az role assignment create --assignee "$APP_ID" --role Contributor --scope "$(az group show -n "$RG" --query id -o tsv)" -o none
az role assignment create --assignee "$APP_ID" --role AcrPush --scope "$ACR_ID" -o none

gh variable set AZURE_CLIENT_ID -b "$APP_ID"
gh variable set AZURE_TENANT_ID -b "$(az account show --query tenantId -o tsv)"
gh variable set AZURE_SUBSCRIPTION_ID -b "$SUB"
gh variable set AZURE_RG -b "$RG"
gh variable set AZURE_ACR -b "$ACR"
gh variable set AZURE_ML_WORKSPACE -b "$AML"
gh variable set AZURE_CONTAINERAPP -b "$APP"

# Budget alert at 80 % of $BUDGET. Not every offer supports budgets, so a failure here is reported, not fatal.
az rest --method put \
  --url "https://management.azure.com/subscriptions/$SUB/providers/Microsoft.Consumption/budgets/vehicle-price?api-version=2023-05-01" \
  --body "{\"properties\": {\"category\": \"Cost\", \"amount\": $BUDGET, \"timeGrain\": \"Monthly\",
    \"timePeriod\": {\"startDate\": \"$(date -u +%Y-%m-01)T00:00:00Z\"},
    \"notifications\": {\"at80\": {\"enabled\": true, \"operator\": \"GreaterThan\", \"threshold\": 80,
      \"contactEmails\": [\"$ALERT_EMAIL\"]}}}}" -o none \
  || echo "WARNING: budget alert not created; set one in the portal (Cost Management) if the offer allows it."

echo "App URL: https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
