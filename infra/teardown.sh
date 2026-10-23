#!/usr/bin/env bash
# Remove everything provision.sh created, so nothing keeps spending credit.
set -euo pipefail

RG=${RG:-rg-vehicle-price}
GH_APP=${GH_APP:-gh-vehicle-price-negotiator}
SUB=$(az account show --query id -o tsv)

az group delete -n "$RG" --yes
APP_ID=$(az ad app list --display-name "$GH_APP" --query "[0].appId" -o tsv)
[ -n "$APP_ID" ] && az ad app delete --id "$APP_ID"
az rest --method delete -o none \
  --url "https://management.azure.com/subscriptions/$SUB/providers/Microsoft.Consumption/budgets/vehicle-price?api-version=2023-05-01" || true
for v in AZURE_CLIENT_ID AZURE_TENANT_ID AZURE_SUBSCRIPTION_ID AZURE_RG AZURE_ACR AZURE_ML_WORKSPACE AZURE_CONTAINERAPP; do
  gh variable delete "$v" || true
done
# The ML workspace and key vault stay soft-deleted (no charge) for up to 14/90 days before Azure purges them.
