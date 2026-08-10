@"
`$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
`$OUT = "finaloutputs\comet_no_flow_compensation_ablation_`$STAMP"
`$LOGDIR = "`$OUT\logs"
New-Item -ItemType Directory -Force -Path `$LOGDIR | Out-Null

"===== COMET NO-FLOW-COMPENSATION ABLATION RUN STARTED at `$(Get-Date) =====" | Tee-Object -FilePath "`$LOGDIR\RUN_STATUS.log"

`$vessels = @(
    @{name="cm_corrected"; path="data/real_vessel_segments_cm_corrected.pkl"},
    @{name="40Left85Right"; path="data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl"},
    @{name="pulmonary_0082_H_PULM_H"; path="data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl"}
)

foreach (`$vessel in `$vessels) {
    `$NET = `$vessel.name
    `$VESSEL = `$vessel.path
    `$RUN_ID = "comet_no_flow_compensation_`${NET}_flows1_3_5"
    `$RUN_DIR = "`$OUT\`$RUN_ID"
    
    "===== STARTING `$RUN_ID at `$(Get-Date) =====" | Tee-Object -FilePath "`$LOGDIR\`$RUN_ID.log" -Append
    
    python -u scripts/run_publication_benchmark.py `
        --output-dir "`$OUT" `
        --run-id "`$RUN_ID" `
        --vessel-file "`$VESSEL" `
        --flow-scales 1,3,5 `
        --n-random-trials 2 `
        --seed 20260611 `
        --controllers comet,comet_no_flow_compensation `
        --steps 2000 `
        --profile spion_carrier_500um `
        --device-mode moving_adaptive_enclosing_gantry `
        --workers 6 `
        2>&1 | Tee-Object -FilePath "`$LOGDIR\`$RUN_ID.log" -Append
    
    "===== BENCHMARK `$RUN_ID FINISHED at `$(Get-Date) =====" | Tee-Object -FilePath "`$LOGDIR\`$RUN_ID.log" -Append
    
    python -u scripts/generate_publication_figures_1mm.py "`$RUN_DIR" `
        2>&1 | Tee-Object -FilePath "`$LOGDIR\`$RUN_ID.log" -Append
}

"===== COMET NO-FLOW-COMPENSATION ABLATION RUN FINISHED at `$(Get-Date). OUT=`$OUT =====" | Tee-Object -FilePath "`$LOGDIR\RUN_STATUS.log" -Append
"@ | Out-File -Encoding UTF8 run_comet_no_flow_compensation_ablation.ps1
