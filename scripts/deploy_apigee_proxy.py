#!/usr/bin/env python3
"""
MS-ADC Apigee Provisioning & Deployment Automation
Provisions the Apigee Pay-as-you-go environment and deploys the MS-ADC-Metrology-v1 proxy bundle.
"""
import io
import os
import sys
import time
import zipfile
import subprocess
import requests
import google.auth
import google.auth.transport.requests

PROJECT_ID = "kshitiz-gemma3"
ORG_NAME = PROJECT_ID
ENV_NAME = "cleanroom-sandbox"
ENVGROUP_NAME = "cleanroom-group"
PROXY_NAME = "MS-ADC-Metrology-v1"
REGION = "us-central1"

def get_token():
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token

def build_proxy_zip():
    buf = io.BytesIO()
    apigee_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apigee"))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(apigee_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, apigee_dir)
                zf.write(full_path, rel_path)
    buf.seek(0)
    return buf.getvalue()

def check_org_status():
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://apigee.googleapis.com/v1/organizations/{ORG_NAME}"
    resp = requests.get(url, headers=headers)
    return resp.status_code == 200, resp.json()

def main():
    print(f"=== Starting Apigee Deployment for Project: {PROJECT_ID} ===")
    token = get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Check Organization
    ready, org_info = check_org_status()
    if not ready:
        print(f"Organization {ORG_NAME} is still provisioning: {org_info}")
        return

    print(f"Organization {ORG_NAME} is ACTIVE: {org_info.get('displayName')}")

    # Create Environment
    env_url = f"https://apigee.googleapis.com/v1/organizations/{ORG_NAME}/environments"
    env_payload = {
        "name": ENV_NAME,
        "description": "Cleanroom Sandbox Environment for MS-ADC Metrology",
        "displayName": "Cleanroom Sandbox",
        "type": "BASE"
    }
    env_resp = requests.post(env_url, headers=headers, json=env_payload)
    print(f"Create Environment Response ({env_resp.status_code}): {env_resp.text[:200]}")

    # Import Proxy Bundle
    zip_bytes = build_proxy_zip()
    import_url = f"https://apigee.googleapis.com/v1/organizations/{ORG_NAME}/apis?action=import&name={PROXY_NAME}"
    upload_headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"}
    import_resp = requests.post(import_url, headers=upload_headers, data=zip_bytes)
    print(f"Import Proxy Response ({import_resp.status_code}): {import_resp.text[:200]}")

    if import_resp.status_code in (200, 201):
        rev = import_resp.json().get("revision", "1")
        # Deploy Proxy Revision
        deploy_url = f"https://apigee.googleapis.com/v1/organizations/{ORG_NAME}/environments/{ENV_NAME}/apis/{PROXY_NAME}/revisions/{rev}/deployments"
        deploy_resp = requests.post(deploy_url, headers=headers)
        print(f"Deploy Proxy Response ({deploy_resp.status_code}): {deploy_resp.text[:200]}")

if __name__ == "__main__":
    main()
