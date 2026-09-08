# ==============================================================================
# MS-ADC Google Cloud Apigee Edge API Gateway Terraform Configuration
# ==============================================================================

# Apigee Pay-As-You-Go Organization
resource "google_apigee_organization" "apigee_org" {
  project_id         = var.project_id
  analytics_region   = var.region
  display_name       = "MS-ADC Cleanroom Metrology Apigee"
  description        = "Enterprise Edge API Gateway for 300mm Silicon Cleanroom Metrology Services"
  runtime_type       = "CLOUD"
  billing_type       = "PAYG"
  authorized_network = google_compute_network.metrology_vpc.id
}

# Apigee Environment Group (Edge Routing Domain)
resource "google_apigee_envgroup" "cleanroom_envgroup" {
  org_id    = google_apigee_organization.apigee_org.id
  name      = "cleanroom-metrology-group"
  hostnames = ["api.cleanroom.fab.internal", "metrology.semicon.cloud"]
}

# Apigee Environment (Cleanroom Runtime Tier)
resource "google_apigee_environment" "cleanroom_env" {
  org_id       = google_apigee_organization.apigee_org.id
  name         = "cleanroom-${var.environment}"
  description  = "Production cleanroom runtime environment for wafer and die metrology inspection"
  display_name = "Cleanroom ${var.environment}"
  type         = "BASE"
}

# Attach Environment to Environment Group
resource "google_apigee_envgroup_attachment" "cleanroom_attachment" {
  envgroup_id = google_apigee_envgroup.cleanroom_envgroup.id
  environment = google_apigee_environment.cleanroom_env.name
}

# Create Zip Archive of Apigee API Proxy Bundle
data "archive_file" "apigee_proxy_bundle" {
  type        = "zip"
  source_dir  = "${path.module}/../apigee"
  output_path = "${path.module}/ms-adc-apigee-bundle.zip"
}

# Deploy Apigee API Proxy Revision
resource "google_apigee_api_proxy" "metrology_proxy" {
  org_id      = google_apigee_organization.apigee_org.id
  name        = "MS-ADC-Metrology-v1"
  bundle      = data.archive_file.apigee_proxy_bundle.output_path
  bundle_hash = data.archive_file.apigee_proxy_bundle.output_base64sha256
}

# Grant Apigee Service Agent Cloud Run Invoker Role
resource "google_cloud_run_v2_service_iam_member" "apigee_cloud_run_invoker" {
  name     = google_cloud_run_v2_service.api_gateway.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-${var.project_id}@gcp-sa-apigee.iam.gserviceaccount.com"
}
