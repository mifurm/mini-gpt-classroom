terraform {
  required_version = ">= 1.6, < 2.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

variable "subscription_id" {
  type        = string
  description = "Explicit Azure subscription in which to deploy the lab website."
}

variable "location" {
  type        = string
  default     = "westeurope"
  description = "Azure region for the Static Web App."
}

provider "azurerm" {
  features {}
  subscription_id                 = var.subscription_id
  resource_provider_registrations = "none"
}

resource "azurerm_resource_group" "website" {
  name     = "rg-mini-gpt-classroom-web"
  location = var.location
  tags = {
    project = "mini-gpt-classroom"
    purpose = "Recorded classroom experiment results"
  }
}

resource "azurerm_static_web_app" "website" {
  name                         = "mini-gpt-classroom-${substr(sha256(var.subscription_id), 0, 8)}"
  resource_group_name          = azurerm_resource_group.website.name
  location                     = azurerm_resource_group.website.location
  sku_tier                     = "Free"
  sku_size                     = "Free"
  preview_environments_enabled = false
  tags                         = azurerm_resource_group.website.tags
}

output "static_web_app_name" {
  value = azurerm_static_web_app.website.name
}

output "resource_group_name" {
  value = azurerm_resource_group.website.name
}

output "website_url" {
  value = "https://${azurerm_static_web_app.website.default_host_name}/"
}
