$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$workspacePath = Split-Path -Parent $PSScriptRoot
$videoDirectory = Join-Path $workspacePath 'data\judge_release\video'
$cues = Get-Content -LiteralPath (Join-Path $videoDirectory 'cues.json') -Raw | ConvertFrom-Json
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($voice.GetInstalledVoices().VoiceInfo.Name -contains 'Microsoft Zira Desktop') { $voice.SelectVoice('Microsoft Zira Desktop') }
$voice.Rate = 0
foreach ($cue in $cues) {
    $voice.SetOutputToWaveFile((Join-Path $videoDirectory ($cue.id + '.wav')))
    $voice.Speak($cue.narration)
    $voice.SetOutputToNull()
}
$voice.Dispose()
