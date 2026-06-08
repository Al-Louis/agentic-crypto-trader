#requires -Version 7.0
<#
.SYNOPSIS
  Provision Apentic dashboard-data hosting on AWS:
    private S3 bucket  ->  dedicated CloudFront distribution on data.alexlouis.dev
    (OAC, managed SimpleCORS, no SPA error fallback)  ->  Route 53 alias.

.DESCRIPTION
  Follows the site's manual-AWS pattern (no Terraform; mirrors .github/workflows/deploy.yml
  and the site's .deploy/ policies). Idempotent and safe to re-run: it reuses an existing
  bucket / OAC / distribution (matched by the `data.alexlouis.dev` alias) instead of creating
  duplicates. Run with an AWS profile that can manage S3 / CloudFront / Route 53 / ACM.

  Does NOT create the IAM publisher user — that's a separate step (printed at the end).

.OUTPUTS
  Writes .deploy/apentic-data.out.json and prints the desktop .env lines.
#>

$ErrorActionPreference = 'Stop'

# ---- config ----------------------------------------------------------------
$Region      = 'us-east-1'
$Account     = '706259162670'
$Bucket      = 'alexlouis-apentic-data'
$Domain      = 'data.alexlouis.dev'
$Apex        = 'alexlouis.dev'
$OacName     = 'apentic-data-oac'
$CachePolicy = '658327ea-f89d-4fab-a63d-7e88639e58f6'   # AWS Managed-CachingOptimized (stable id)
$RespPolicy  = '60669652-455b-4ae9-85a4-c4c02393f86c'   # AWS Managed-SimpleCORS       (stable id)
$CfAliasZone = 'Z2FDTNDATAQYW2'                           # CloudFront alias hosted-zone (global const)
$OutFile     = Join-Path $PSScriptRoot 'apentic-data.out.json'

# ---- helpers ---------------------------------------------------------------
function Aws {
    # Run aws, throw on non-zero exit, return stdout as one string (for ConvertFrom-Json / .Trim()).
    $out = & aws @args
    if ($LASTEXITCODE -ne 0) { throw "FAILED: aws $($args -join ' ') (exit $LASTEXITCODE)" }
    return ($out -join "`n")
}
function Write-JsonFile([object]$Obj, [string]$Name) {
    $path = Join-Path $env:TEMP $Name
    $Obj | ConvertTo-Json -Depth 16 | Set-Content -Path $path -Encoding utf8
    return "file://$($path -replace '\\','/')"   # forward slashes: most portable for aws file://
}

# ---- 1. private S3 bucket --------------------------------------------------
Write-Host "==> S3 bucket $Bucket"
& aws s3api head-bucket --bucket $Bucket 2>$null
if ($LASTEXITCODE -ne 0) {
    Aws s3api create-bucket --bucket $Bucket --region $Region | Out-Null
    Aws s3api put-public-access-block --bucket $Bucket --public-access-block-configuration `
        'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true' | Out-Null
    Write-Host "    created (private)"
} else { Write-Host "    exists - skipping create" }

# ---- 2. ACM cert (*.alexlouis.dev, ISSUED, us-east-1) ----------------------
Write-Host "==> ACM cert *.$Apex"
$CertArn = (Aws acm list-certificates --region $Region --certificate-statuses ISSUED `
    --query "CertificateSummaryList[?DomainName=='*.$Apex'].CertificateArn" --output text).Trim()
if (-not $CertArn -or $CertArn -eq 'None') {
    throw "No ISSUED *.$Apex certificate in $Region. Request + DNS-validate it first."
}
Write-Host "    $CertArn"

# ---- 3. Origin Access Control (reuse by name) ------------------------------
Write-Host "==> Origin Access Control $OacName"
$OacId = (Aws cloudfront list-origin-access-controls `
    --query "OriginAccessControlList.Items[?Name=='$OacName'].Id | [0]" --output text).Trim()
if (-not $OacId -or $OacId -eq 'None') {
    $OacId = (Aws cloudfront create-origin-access-control --origin-access-control-config `
        "Name=$OacName,SigningProtocol=sigv4,SigningBehavior=always,OriginAccessControlOriginType=s3" `
        --query "OriginAccessControl.Id" --output text).Trim()
    Write-Host "    created $OacId"
} else { Write-Host "    exists $OacId" }

# ---- 4. CloudFront distribution (reuse by alias) ---------------------------
Write-Host "==> CloudFront distribution for $Domain"
$DistId = (Aws cloudfront list-distributions `
    --query "DistributionList.Items[?contains(Aliases.Items, '$Domain')].Id | [0]" --output text).Trim()
if (-not $DistId -or $DistId -eq 'None') {
    $cfg = [ordered]@{
        CallerReference   = [guid]::NewGuid().ToString()
        Aliases           = @{ Quantity = 1; Items = @($Domain) }
        DefaultRootObject = ''
        Origins           = @{ Quantity = 1; Items = @(@{
                Id                    = "s3-$Bucket"
                DomainName            = "$Bucket.s3.$Region.amazonaws.com"
                OriginPath            = ''
                CustomHeaders         = @{ Quantity = 0 }
                S3OriginConfig        = @{ OriginAccessIdentity = '' }   # empty + OAC id below = OAC, not legacy OAI
                OriginAccessControlId = $OacId
                ConnectionAttempts    = 3
                ConnectionTimeout     = 10
                OriginShield          = @{ Enabled = $false }
            }) }
        DefaultCacheBehavior = [ordered]@{
            TargetOriginId             = "s3-$Bucket"
            ViewerProtocolPolicy       = 'redirect-to-https'
            AllowedMethods             = @{ Quantity = 2; Items = @('GET', 'HEAD'); CachedMethods = @{ Quantity = 2; Items = @('GET', 'HEAD') } }
            Compress                   = $true
            CachePolicyId              = $CachePolicy        # replaces legacy ForwardedValues
            ResponseHeadersPolicyId    = $RespPolicy         # SimpleCORS -> Access-Control-Allow-Origin
            SmoothStreaming            = $false
            FieldLevelEncryptionId     = ''
            LambdaFunctionAssociations = @{ Quantity = 0 }
            FunctionAssociations       = @{ Quantity = 0 }
        }
        CacheBehaviors       = @{ Quantity = 0 }
        CustomErrorResponses = @{ Quantity = 0 }            # no SPA fallback -> clean 403/404 for data
        Comment              = "Apentic data ($Domain)"
        Logging              = @{ Enabled = $false; IncludeCookies = $false; Bucket = ''; Prefix = '' }
        PriceClass           = 'PriceClass_100'
        Enabled              = $true
        ViewerCertificate    = @{ ACMCertificateArn = $CertArn; SSLSupportMethod = 'sni-only'; MinimumProtocolVersion = 'TLSv1.2_2021' }
        Restrictions         = @{ GeoRestriction = @{ RestrictionType = 'none'; Quantity = 0 } }
        WebACLId             = ''
        HttpVersion          = 'http2and3'
        IsIPV6Enabled        = $true
    }
    $cfgUri = Write-JsonFile $cfg 'apentic-dist.json'
    $resp = (Aws cloudfront create-distribution --distribution-config $cfgUri --output json) | ConvertFrom-Json
    $DistId = $resp.Distribution.Id
    $DistDomain = $resp.Distribution.DomainName
    Write-Host "    created $DistId ($DistDomain)"
} else {
    $DistDomain = (Aws cloudfront get-distribution --id $DistId --query "Distribution.DomainName" --output text).Trim()
    Write-Host "    exists $DistId ($DistDomain) - skipping create"
}

# ---- 5. bucket policy: only this distribution may read the private bucket ---
Write-Host "==> Bucket policy (OAC read)"
$policy = @{
    Version   = '2012-10-17'
    Statement = @(@{
            Sid       = 'AllowCloudFrontServicePrincipalRead'
            Effect    = 'Allow'
            Principal = @{ Service = 'cloudfront.amazonaws.com' }
            Action    = 's3:GetObject'
            Resource  = "arn:aws:s3:::$Bucket/*"
            Condition = @{ StringEquals = @{ 'AWS:SourceArn' = "arn:aws:cloudfront::${Account}:distribution/$DistId" } }
        })
}
$polUri = Write-JsonFile $policy 'apentic-bucket-policy.json'
Aws s3api put-bucket-policy --bucket $Bucket --policy $polUri | Out-Null
Write-Host "    applied"

# ---- 6. Route 53 alias -> the distribution ---------------------------------
Write-Host "==> Route 53 alias $Domain"
$ZoneId = ((Aws route53 list-hosted-zones-by-name --dns-name $Apex --query "HostedZones[0].Id" --output text) -replace '/hostedzone/', '').Trim()
$alias = @{ HostedZoneId = $CfAliasZone; DNSName = $DistDomain; EvaluateTargetHealth = $false }
$batch = @{
    Comment = "$Domain -> Apentic CloudFront"
    Changes = @(
        @{ Action = 'UPSERT'; ResourceRecordSet = @{ Name = $Domain; Type = 'A'; AliasTarget = $alias } }
        @{ Action = 'UPSERT'; ResourceRecordSet = @{ Name = $Domain; Type = 'AAAA'; AliasTarget = $alias } }
    )
}
$batchUri = Write-JsonFile $batch 'apentic-r53.json'
Aws route53 change-resource-record-sets --hosted-zone-id $ZoneId --change-batch $batchUri | Out-Null
Write-Host "    upserted in zone $ZoneId"

# ---- 7. wait for deploy, record, and print next steps ----------------------
Write-Host "==> Waiting for distribution to deploy (a few minutes)..."
Aws cloudfront wait distribution-deployed --id $DistId | Out-Null

[ordered]@{
    bucket             = $Bucket
    domain             = $Domain
    distributionId     = $DistId
    distributionDomain = $DistDomain
    oacId              = $OacId
    certArn            = $CertArn
} | ConvertTo-Json | Set-Content -Path $OutFile -Encoding utf8

Write-Host ""
Write-Host "DONE. Recorded -> $OutFile"
Write-Host ""
Write-Host "Desktop .env (training host):"
Write-Host "  APENTIC_PUBLISH_TARGET=s3://$Bucket"
Write-Host "  APENTIC_CLOUDFRONT_DIST_ID=$DistId"
Write-Host ""
Write-Host "Next (separate, manual): create the scoped IAM publisher user for the desktop:"
Write-Host "  s3:PutObject/GetObject on arn:aws:s3:::$Bucket/*"
Write-Host "  cloudfront:CreateInvalidation on arn:aws:cloudfront::${Account}:distribution/$DistId"
Write-Host "Frontend: PUBLIC_APENTIC_DATA=https://$Domain (and remove the committed public/apentic/data samples)."
