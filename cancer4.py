#!/usr/bin/env python3
"""
Breast Health Report Companion — educational report explainer.
Run:
    source venv/bin/activate
    pip install flask numpy scikit-learn  # required for the optional ML demo
# Optional report upload: pip install pypdf pillow pytesseract
    python3 cancer3.py
Open http://127.0.0.1:5000

Optional PDF text extraction:
    pip install pypdf

Optional image OCR (also install the system Tesseract OCR program):
    pip install pillow pytesseract
This app explains words explicitly present in a report. It does NOT diagnose cancer,
calculate a patient's cancer probability, or replace a pathologist/doctor.
"""
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime
from pathlib import Path
import json
import re
import uuid


# ---------------------------------------------------------------------------
# Optional educational machine-learning demo from the original cancer.py.
# This model uses the built-in Wisconsin dataset and expects 30 digitised
# cell-nucleus measurements. It must NOT be used to interpret pasted reports.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_score, recall_score, roc_auc_score)
    from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                         train_test_split)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    DATASET = load_breast_cancer()
    MODEL_X, MODEL_Y = DATASET.data, DATASET.target
    MODEL_NAMES = [n.title() for n in DATASET.feature_names]
    XTR, XTE, YTR, YTE = train_test_split(
        MODEL_X, MODEL_Y, test_size=0.2, stratify=MODEL_Y, random_state=42
    )
    LR_MODEL = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))
    ENSEMBLE = VotingClassifier([
        ("Logistic regression", LR_MODEL),
        ("SVM", make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=0))),
        ("Random forest", RandomForestClassifier(200, random_state=0))
    ], voting="soft").fit(XTR, YTR)

    model_prob = ENSEMBLE.predict_proba(XTE)[:, 0]
    model_pred = (model_prob >= .5).astype(int)
    model_truth = (YTE == 0).astype(int)
    MODEL_METRICS = {
        "accuracy": float(accuracy_score(model_truth, model_pred)),
        "cv_accuracy": float(cross_val_score(
            ENSEMBLE, MODEL_X, MODEL_Y,
            cv=StratifiedKFold(5, shuffle=True, random_state=1)
        ).mean()),
        "recall": float(recall_score(model_truth, model_pred)),
        "precision": float(precision_score(model_truth, model_pred)),
        "auc": float(roc_auc_score(model_truth, model_prob)),
        "n_test": int(len(YTE)),
        "confusion": confusion_matrix(model_truth, model_pred).tolist()
    }
    MODEL_FEATURES = [{
        "name": MODEL_NAMES[i],
        "min": float(MODEL_X[:, i].min()),
        "max": float(MODEL_X[:, i].max()),
        "benign": float(np.median(MODEL_X[MODEL_Y == 1, i])),
        "malignant": float(np.median(MODEL_X[MODEL_Y == 0, i])),
        "group": ["Average", "Variation", "Worst"][i // 10]
    } for i in range(30)]
    fitted_logistic = ENSEMBLE.named_estimators_["Logistic regression"]
    MODEL_SCALER = fitted_logistic.named_steps["standardscaler"]
    MODEL_CLF = fitted_logistic.named_steps["logisticregression"]
    MODEL_RNG = np.random.default_rng(0)
    MODEL_READY = True
    print("Educational ML demo loaded:", MODEL_METRICS)
except Exception as model_error:
    MODEL_READY = False
    MODEL_ERROR = str(model_error)
    print("Optional ML demo unavailable:", MODEL_ERROR)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
DATA_FILE = Path(__file__).with_name("patient_progress.json")

PAGE = r"""
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breast Health Report Companion</title>
<style>
:root{--ink:#17343b;--muted:#647982;--teal:#087e80;--pale:#eaf5f3;--line:#dce7e8;--bg:#f4f8f8;--white:#fff;--amber:#9b6200;--amberbg:#fff5df;--red:#a92d40;--redbg:#fff0f1;--green:#176c50;--greenbg:#eaf7f0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,Segoe UI,Arial,sans-serif}
header{background:linear-gradient(120deg,#123a43,#087e80);color:white;padding:30px 20px}
.wrap{max-width:1120px;margin:auto;padding:0 18px}h1{font-size:clamp(26px,4vw,38px);margin:0 0 6px}header p{margin:0;color:#d9eeee;max-width:760px}
.notice{background:#fff4db;color:#684800;padding:11px 18px;text-align:center;border-bottom:1px solid #ead6a7;font-size:14px}
nav{display:flex;gap:8px;flex-wrap:wrap;padding:18px 0 12px}
nav button,.btn{border:1px solid var(--line);background:white;color:var(--ink);padding:10px 14px;border-radius:10px;font-weight:650;cursor:pointer}
nav button.active,.btn.primary{background:var(--teal);color:white;border-color:var(--teal)}
main{padding-bottom:38px}.tab{display:none}.tab.active{display:block}
.card{background:white;border:1px solid var(--line);border-radius:16px;padding:20px;margin:0 0 15px;box-shadow:0 5px 18px #183c4308}
h2{font-size:21px;margin:0 0 7px}h3{font-size:16px;margin:0 0 5px}.muted{color:var(--muted)}.small{font-size:13px;color:var(--muted)}
.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:16px;align-items:start}@media(max-width:820px){.grid{grid-template-columns:1fr}}
textarea,input,select{width:100%;padding:12px;border:1px solid #cbdadc;border-radius:10px;font:inherit;color:var(--ink);background:white}
textarea{min-height:230px;resize:vertical}.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.row>*{flex:1}
.btn{display:inline-block;margin-top:10px}.btn:disabled{opacity:.55;cursor:not-allowed}
.result{border-radius:12px;padding:15px;margin-top:12px}.green{background:var(--greenbg);border:1px solid #b8dfc9}.amber{background:var(--amberbg);border:1px solid #ead39a}.red{background:var(--redbg);border:1px solid #efc0c7}.neutral{background:var(--pale);border:1px solid #c8e4df}
.pills{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.pill{padding:5px 10px;border-radius:30px;background:#edf3f4;font-size:13px}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}@media(max-width:650px){.steps{grid-template-columns:1fr}}
.step{background:#f3f8f8;border-radius:12px;padding:13px;text-align:center}.step .num{display:grid;place-items:center;margin:0 auto 8px;width:34px;height:34px;border-radius:50%;background:var(--teal);color:white;font-weight:700}
.illustrations{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.illustrations figure{margin:0;background:#f7fafa;border:1px solid var(--line);border-radius:12px;padding:10px;text-align:center}.illustrations svg{width:100%;max-height:170px}.illustrations figcaption{font-size:13px;color:var(--muted)}
ul{padding-left:21px}li{margin:5px 0}.hidden{display:none!important}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid var(--line);vertical-align:top}
.alert{padding:12px;border-radius:10px;background:#fff4db;border:1px solid #ead6a7;margin-top:10px}
footer{padding:20px 0 34px;color:var(--muted);font-size:12px}
</style></head>
<body>
<header><div class="wrap"><h1>Breast Health Report Companion</h1><p>Understand the wording in a breast report, learn what questions to ask, and keep a simple record of follow-up visits.</p></div></header>
<div class="notice"><b>Educational information only.</b> This tool cannot diagnose breast cancer or calculate an individual patient's cancer risk. A qualified clinician must interpret the complete report.</div>
<div class="wrap"><nav>
<button class="active" data-tab="analyze">Understand my report</button>
<button data-tab="learn">Pictures & learning</button>
<button data-tab="symptoms">Symptoms & when to seek help</button>
<button data-tab="progress">Patient progress</button>
<button data-tab="precautions">Precautions</button>
<button data-tab="model">Educational ML demo</button>
</nav>
<main>
<section id="analyze" class="tab active">
<div class="grid">
<div class="card">
<h2>Paste the report text</h2><p class="muted">Copy the wording from the biopsy, histopathology, cytology, ultrasound or mammogram report. Remove the patient's name, address, phone number and hospital ID first.</p>
<textarea id="report" placeholder="Paste the report text here...&#10;&#10;Example: Histopathology: fibroadenoma. No evidence of malignancy."></textarea>
<div class="row"><button class="btn primary" id="analyzeBtn">Explain this report</button><label class="btn" style="text-align:center;cursor:pointer">Upload PDF / text / image<input id="file" type="file" accept=".pdf,.txt,.png,.jpg,.jpeg" style="display:none"></label></div>
<p class="small" id="fileStatus">PDF text extraction needs the optional pypdf package. Image reading needs optional OCR setup.</p>
</div>
<div class="card">
<h2>How to use it</h2>
<div class="steps"><div class="step"><div class="num">1</div><b>Copy</b><p class="small">Copy report wording</p></div><div class="step"><div class="num">2</div><b>Paste</b><p class="small">Add it in the box</p></div><div class="step"><div class="num">3</div><b>Review</b><p class="small">Read explanation and discuss with doctor</p></div></div>
<div class="alert"><b>Important:</b> A report may describe imaging suspicion, a biopsy result, or a final diagnosis. These are not interchangeable. This tool will show when the wording is unclear or insufficient.</div>
</div>
</div>
<div id="analysisResult" class="card hidden" aria-live="polite"></div>
</section>

<section id="learn" class="tab">
<div class="card"><h2>Breast health: simple picture guide</h2><p class="muted">These are simplified educational drawings, not scans or photographs of a real patient's tumour.</p>
<div class="illustrations">
<figure><svg viewBox="0 0 220 150" role="img" aria-label="Simplified breast cross-section"><path d="M20 80 Q75 8 175 70 Q195 85 178 105 Q90 140 20 80" fill="#f3c9b9" stroke="#a86f62" stroke-width="3"/><path d="M35 78 Q85 30 160 75" fill="none" stroke="#d99b8a" stroke-width="8" stroke-linecap="round"/><g fill="#e7a44e" stroke="#b77b35"><circle cx="85" cy="69" r="10"/><circle cx="112" cy="84" r="12"/><circle cx="139" cy="69" r="9"/></g><path d="M179 86 l24 0" stroke="#a86f62" stroke-width="4"/><text x="110" y="143" text-anchor="middle" font-size="10" fill="#52666c">Illustrative anatomy</text></svg><figcaption><b>Breast tissue</b><br>Contains ducts, lobules, fat and connective tissue.</figcaption></figure>
<figure><svg viewBox="0 0 220 150" role="img" aria-label="Illustration of a circumscribed lump"><path d="M15 80 Q75 15 180 75 Q198 90 175 110 Q85 140 15 80" fill="#f3c9b9" stroke="#a86f62" stroke-width="3"/><circle cx="108" cy="78" r="23" fill="#e8b56d" stroke="#9c6c2d" stroke-width="3"/><path d="M108 48v-15m0 0l-5 7m5-7l5 7" stroke="#9c6c2d" stroke-width="2" fill="none"/><text x="110" y="143" text-anchor="middle" font-size="10" fill="#52666c">Not to scale</text></svg><figcaption><b>Well-defined lump</b><br>Some non-cancerous conditions can form lumps; examination is still important.</figcaption></figure>
<figure><svg viewBox="0 0 220 150" role="img" aria-label="Illustration of irregular lesion"><path d="M15 80 Q75 15 180 75 Q198 90 175 110 Q85 140 15 80" fill="#f3c9b9" stroke="#a86f62" stroke-width="3"/><path d="M108 48l10 9 15-2 2 14 11 10-10 11-2 14-15-4-12 9-8-12-14-3 5-14-6-12 14-5z" fill="#bd6870" stroke="#8d3c49" stroke-width="3"/><text x="110" y="143" text-anchor="middle" font-size="10" fill="#52666c">Not to scale</text></svg><figcaption><b>Irregular appearance</b><br>Imaging appearance alone does not confirm cancer; biopsy may be needed.</figcaption></figure>
<figure><svg viewBox="0 0 220 150" role="img" aria-label="Illustration of a biopsy sample"><rect x="35" y="25" width="150" height="95" rx="12" fill="#f7e7d9" stroke="#b98b75" stroke-width="3"/><g fill="#b46e7d" opacity=".8"><circle cx="72" cy="60" r="11"/><circle cx="105" cy="52" r="8"/><circle cx="140" cy="70" r="13"/><circle cx="91" cy="91" r="7"/><circle cx="151" cy="100" r="8"/></g><text x="110" y="143" text-anchor="middle" font-size="10" fill="#52666c">Conceptual tissue sample</text></svg><figcaption><b>Biopsy</b><br>A pathologist examines sampled tissue to identify what it is.</figcaption></figure>
</div></div>
<div class="card"><h2>Useful report words</h2><ul><li><b>Benign:</b> non-cancerous.</li><li><b>Malignant:</b> cancerous.</li><li><b>In situ:</b> abnormal cells remain within the structure where they began; the exact diagnosis matters.</li><li><b>Invasive:</b> cancer cells have grown beyond their original lining or boundary.</li><li><b>Indeterminate / suspicious:</b> the wording does not establish a definite benign or malignant diagnosis; ask the treating clinician what further assessment is needed.</li><li><b>Negative for malignancy:</b> no cancer was identified in the examined sample; the doctor must check that the sample and findings fit the imaging and clinical picture.</li></ul></div>
</section>

<section id="symptoms" class="tab">
<div class="card"><h2>Changes worth getting checked</h2><p class="muted">Tick any changes the patient has noticed. This checklist cannot diagnose cancer.</p>
<div id="symptomList"></div><div id="symptomAdvice" class="result neutral">If a new or persistent breast change is present, arrange a clinical assessment.</div></div>
<div class="card"><h2>Seek urgent medical help</h2><p>Get prompt medical advice for a new breast lump, new nipple inversion, bloody nipple discharge, skin dimpling, or a persistent change in breast shape or skin. A hot, red, painful breast with fever or feeling very unwell needs same-day medical assessment because infection may need treatment.</p><p class="small">If there is severe illness or rapidly worsening symptoms, use local emergency services.</p></div>
</section>

<section id="progress" class="tab">
<div class="card"><h2>Patient follow-up record</h2><p class="muted">This demo stores only the patient label, visit date and a short note in a local JSON file beside this program. Do not enter names or identifying details on a shared computer.</p>
<div class="row"><input id="patientId" placeholder="Patient code (avoid full name)"><input id="visitDate" type="date"></div>
<textarea id="visitNote" style="min-height:90px;margin-top:10px" placeholder="Example: Report discussed with breast specialist; follow-up scan planned. Do not paste the full report here."></textarea>
<button class="btn primary" id="saveVisit">Save follow-up note</button><p id="saveStatus" class="small"></p><div id="history"></div>
</div></section>

<section id="precautions" class="tab">
<div class="card"><h2>Helpful precautions</h2><ul><li>Keep copies of imaging, biopsy and pathology reports and bring them to appointments.</li><li>Attend the follow-up recommended by the treating clinician.</li><li>Ask whether the biopsy result matches the imaging findings (often called radiology-pathology concordance).</li><li>Ask the doctor to explain unfamiliar terms, the exact diagnosis and whether more tests are needed.</li><li>Do not start, stop or change medicines or cancer treatment based on this website.</li><li>Do not rely on supplements, special diets or home remedies to treat a breast lump or cancer.</li><li>Healthy habits such as regular physical activity, avoiding tobacco and limiting alcohol support general health but do not replace screening or treatment.</li></ul></div>
<div class="card"><h2>Questions to take to the doctor</h2><ol><li>What is the exact diagnosis written in my report?</li><li>Is this a confirmed biopsy diagnosis or an imaging suspicion?</li><li>Does the result match the scan and examination?</li><li>Do I need another test, a repeat biopsy or treatment?</li><li>When should I return, and which symptoms should make me come sooner?</li></ol></div>
</section>

<section id="model" class="tab">
<div class="card">
<h2>Educational cell-measurement demo</h2>
<div class="alert"><b>Not for patient diagnosis.</b> This separate demonstration uses 30 numerical features from a public research dataset. These values are not normally present in a patient's written biopsy report. Do not guess or manually invent them, and do not use this demo to decide care.</div>
<div id="modelUnavailable" class="result amber hidden"></div>
<div id="modelControls">
<p class="muted">Use the presets only to explore how the research model responds to sample data.</p>
<div class="row">
<button class="btn" id="mBenign">Typical benign example</button>
<button class="btn" id="mMalignant">Typical malignant example</button>
<button class="btn" id="mRandomBenign">Random dataset benign</button>
<button class="btn" id="mRandomMalignant">Random dataset malignant</button>
</div>
<div id="mSliders"></div>
<button class="btn primary" id="mPredict">Run educational demo</button>
<div id="mOutput" class="result neutral hidden"></div>
</div>
</div>
</section>

</main>
<footer>Privacy: this app runs on your computer. The report text is analysed in the browser/server process and is not sent to an AI service. Follow-up notes are stored locally in patient_progress.json. This is a learning project, not a certified clinical system.</footer>
</div>
<script>
const $=id=>document.getElementById(id);
document.querySelectorAll('nav button').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('nav button,.tab').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.tab).classList.add('active');if(b.dataset.tab==='progress')loadHistory()}));
$('analyzeBtn').onclick=async()=>{const text=$('report').value.trim();if(!text){alert('Paste the report text first.');return}const b=$('analyzeBtn');b.disabled=true;b.textContent='Reading report…';try{const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({report:text})});const d=await r.json();showAnalysis(d)}catch(e){alert('Could not analyse the text. Please try again.')}finally{b.disabled=false;b.textContent='Explain this report'}};
function showAnalysis(d){const box=$('analysisResult');box.classList.remove('hidden');box.className='card';box.innerHTML=`<h2>What your report says</h2><div class="result ${d.tone}"><b>${esc(d.heading)}</b><p>${esc(d.summary)}</p></div><h3 style="margin-top:16px">Words found in the report</h3><div class="pills">${d.terms.length?d.terms.map(x=>`<span class="pill">${esc(x)}</span>`).join(''):'<span class="pill">No recognised diagnosis terms found</span>'}</div><h3>What this means</h3><ul>${d.explanations.map(x=>`<li>${esc(x)}</li>`).join('')}</ul><div class="alert"><b>Next step:</b> ${esc(d.next)}</div><p class="small" style="margin-top:10px">${esc(d.caution)}</p><button class="btn" onclick="window.print()">Print explanation</button>`;box.scrollIntoView({behavior:'smooth',block:'start'})}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
$('file').onchange=async e=>{const f=e.target.files[0];if(!f)return;const fd=new FormData();fd.append('file',f);$('fileStatus').textContent='Reading '+f.name+'…';try{const r=await fetch('/extract',{method:'POST',body:fd});const d=await r.json();if(!r.ok)throw new Error(d.error);$('report').value=d.text;$('fileStatus').textContent='Text extracted. Review it, then click Explain this report.'}catch(err){$('fileStatus').textContent=err.message}e.target.value=''};
const symptoms=['New lump or thickening in the breast or armpit','Change in breast size or shape','Skin dimpling, puckering or orange-peel texture','New nipple turning inward','Bloody or unusual nipple discharge','Persistent redness, rash or scaling around the nipple','Persistent pain in one area'];
$('symptomList').innerHTML=symptoms.map((s,i)=>`<label style="display:block;padding:8px;border-bottom:1px solid var(--line)"><input type="checkbox" style="width:auto;margin-right:8px" data-sym="${i}">${s}</label>`).join('');
$('symptomList').onchange=()=>{const n=$('symptomList').querySelectorAll('input:checked').length;const a=$('symptomAdvice');a.className='result '+(n?'amber':'neutral');a.textContent=n?'One or more changes selected. Please arrange a clinical assessment promptly; this checklist cannot tell whether the cause is cancer.':'No changes selected. If a new or persistent change appears, seek medical advice.'};
$('visitDate').value=new Date().toISOString().slice(0,10);
$('saveVisit').onclick=async()=>{const id=$('patientId').value.trim(),note=$('visitNote').value.trim();if(!id||!note){alert('Enter a patient code and a short follow-up note.');return}const r=await fetch('/progress',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({patient_id:id,date:$('visitDate').value,note})});const d=await r.json();$('saveStatus').textContent=d.message; $('visitNote').value='';loadHistory()};
async function loadHistory(){const id=$('patientId').value.trim();const r=await fetch('/progress'+(id?'?patient_id='+encodeURIComponent(id):''));const d=await r.json();$('history').innerHTML=d.records.length?'<h3 style="margin-top:18px">Saved visits</h3><table><tr><th>Date</th><th>Patient code</th><th>Note</th></tr>'+d.records.map(x=>`<tr><td>${esc(x.date)}</td><td>${esc(x.patient_id)}</td><td>${esc(x.note)}</td></tr>`).join('')+'</table>':'<p class="small" style="margin-top:15px">No saved visits for this patient code yet.</p>'}

// Educational ML tab: model output is deliberately separated from report analysis.
(async function initModelDemo(){
 const unavailable=$('modelUnavailable'), controls=$('modelControls'), sliderBox=$('mSliders');
 try{
  const metaRes=await fetch('/model/meta'); const meta=await metaRes.json();
  if(!metaRes.ok) throw new Error(meta.error||'Model unavailable');
  let features=meta.features, values=features.map(f=>f.benign);
  sliderBox.innerHTML=['Average','Variation','Worst'].map((group,g)=>`<details ${g===0?'open':''}><summary>${group} measurements</summary><div class="grid">${features.map((f,i)=>({f,i})).filter(x=>x.f.group===group).map(({f,i})=>`<label style="display:block;margin:9px 0"><span>${esc(f.name)}: <b id="mv${i}">${Number(values[i]).toPrecision(4)}</b></span><input type="range" id="ms${i}" min="${f.min}" max="${f.max}" step="${(f.max-f.min)/400}" value="${values[i]}"></label>`).join('')}</div></details>`).join('');
  features.forEach((f,i)=>{$('ms'+i).oninput=e=>{values[i]=Number(e.target.value);$('mv'+i).textContent=values[i].toPrecision(4)}});
  const setPreset=v=>{values=v;features.forEach((f,i)=>{$('ms'+i).value=v[i];$('mv'+i).textContent=v[i].toPrecision(4)});$('mOutput').classList.add('hidden')};
  $('mBenign').onclick=()=>setPreset(features.map(f=>f.benign));
  $('mMalignant').onclick=()=>setPreset(features.map(f=>f.malignant));
  $('mRandomBenign').onclick=async()=>{const d=await(await fetch('/model/sample?cls=benign')).json();setPreset(d.values)};
  $('mRandomMalignant').onclick=async()=>{const d=await(await fetch('/model/sample?cls=malignant')).json();setPreset(d.values)};
  $('mPredict').onclick=async()=>{const r=await fetch('/model/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({values})});const d=await r.json();const out=$('mOutput');out.classList.remove('hidden');out.className='result neutral';out.innerHTML=r.ok?`<b>Research model estimate: ${(d.estimate*100).toFixed(1)}%</b><p>${esc(d.note)}</p>`:esc(d.error)};
 }catch(e){controls.classList.add('hidden');unavailable.classList.remove('hidden');unavailable.textContent='The report analyser still works. The optional original machine-learning demonstration needs its packages: '+e.message}
})();

</script></body></html>
"""

def classify_report(text):
    """Conservative keyword explanation; no diagnosis or probability is generated."""
    t = re.sub(r"\s+", " ", text.lower())
    terms, explanations = [], []
    # Detect negated mentions before positive cancer terms.
    benign = bool(re.search(r"\b(benign|fibroadenoma|fibrocystic|cystic change|usual ductal hyperplasia)\b", t))
    malignant = bool(re.search(r"\b(malignant|carcinoma|adenocarcinoma|invasive ductal|invasive lobular)\b", t))
    negative = bool(re.search(r"\b(no evidence of malignancy|negative for malignancy|negative for cancer|no malignancy identified|absence of malignancy)\b", t))
    uncertain = bool(re.search(r"\b(suspicious|cannot exclude|cannot rule out|indeterminate|atypia|atypical|inconclusive|insufficient|inadequate)\b", t))
    imaging = bool(re.search(r"\b(birads|bi-rads|birads)\b", t))
    biopsy = bool(re.search(r"\b(biopsy|histopathology|histopathological|core needle|fine needle|cytology|pathology)\b", t))
    if negative:
        terms.append("No malignancy identified / negative for malignancy")
        explanations.append("The text says that malignancy was not identified in the examined sample.")
    if benign:
        terms.append("Benign / non-cancerous wording")
        explanations.append("The report contains wording associated with a benign (non-cancerous) finding. The exact named condition and the full report matter.")
    if malignant and not negative:
        terms.append("Malignant / carcinoma wording")
        explanations.append("The report contains a term used for cancer. The exact type, grade, extent and complete pathology findings must be reviewed by the treating specialist.")
    if uncertain:
        terms.append("Uncertain / suspicious wording")
        explanations.append("The report includes wording that may mean the result is not definitive or needs further assessment. Ask the doctor what additional test or biopsy is needed.")
    if imaging:
        terms.append("BI-RADS imaging category")
        m = re.search(r"\bbi-?rads?\s*(?:category|cat)?\s*([0-6])\b", t)
        if m:
            explanations.append("A BI-RADS category appears in the text: " + m.group(1) + ". This is an imaging assessment category, not a biopsy diagnosis. Ask the radiologist/doctor to explain the category and recommended next step.")
        else:
            explanations.append("BI-RADS is an imaging reporting system. The category number was not clearly identified in the pasted text.")
    if biopsy:
        terms.append("Biopsy / pathology wording")
        explanations.append("The report appears to refer to tissue or cell examination. A pathologist's final diagnosis should be read exactly as written.")
    if re.search(r"\b(in situ|dcis|lcis)\b", t):
        terms.append("In-situ wording")
        explanations.append("The report mentions an in-situ term. This has a specific medical meaning and should be discussed with the breast specialist; it should not be simplified to an ordinary benign lump.")
    if re.search(r"\binvasive\b", t):
        terms.append("Invasive wording")
        explanations.append("The report uses the word invasive. Ask the specialist to explain the exact cancer type and what further information is needed.")
    if re.search(r"\bher2\b|\berbb2\b", t):
        terms.append("HER2")
        explanations.append("HER2 is a biomarker that may help guide treatment if cancer is diagnosed. The result (positive, negative or equivocal) and any confirmatory test matter.")
    if re.search(r"\bestrogen receptor\b|\ber\b|\bprogesterone receptor\b|\bpr\b", t):
        terms.append("Hormone receptor markers")
        explanations.append("The report may mention hormone receptors. Their actual results and percentages should be interpreted by the treating team.")
    if not terms:
        heading = "Not enough clear information to interpret"
        summary = "The pasted text does not contain clear report terms this simple tool can safely explain."
        tone = "amber"
        explanations.append("Check that you pasted the final impression or diagnosis section, not only appointment details or a scan description.")
        next_step = "Show the complete report to the doctor who ordered the test. Do not infer a diagnosis from this tool."
    elif uncertain:
        heading = "The report wording needs medical clarification"
        summary = "The text includes uncertain, suspicious or atypical wording. This is not the same as a confirmed cancer diagnosis."
        tone = "amber"
        next_step = "Contact the treating doctor or breast clinic promptly to clarify whether more imaging, repeat sampling or another test is required."
    elif malignant and not negative:
        heading = "Cancer-related wording is present"
        summary = "The report includes malignant/carcinoma wording. This tool cannot confirm the diagnosis or determine its stage; a clinician must review the complete pathology report."
        tone = "red"
        next_step = "Arrange prompt review with the treating doctor or breast specialist. Bring the complete report and imaging."
    elif negative or benign:
        heading = "The report contains benign / negative wording"
        summary = "The text includes wording that may indicate a non-cancerous finding or no malignancy in the examined sample. The doctor must check that the sample result matches the scan and examination."
        tone = "green"
        next_step = "Discuss the final report and recommended follow-up with the doctor. Seek review if symptoms persist or change."
    else:
        heading = "Report terms detected — clinician review needed"
        summary = "Some medical terms were found, but they do not establish a clear benign or malignant conclusion."
        tone = "amber"
        next_step = "Ask the doctor to explain the final impression and whether additional assessment is needed."
    caution = ("This is a basic keyword explainer, not an AI medical diagnosis. It may miss context, negation or errors in copied text. "
               "A negative sample does not always rule out disease if the sample is inadequate or does not match imaging. "
               "Only the treating clinician/pathologist can interpret the complete case.")
    return {"heading": heading, "summary": summary, "tone": tone, "terms": terms,
            "explanations": explanations, "next": next_step, "caution": caution}

def load_records():
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []

def save_records(records):
    DATA_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")

@app.get("/")
def home():
    return render_template_string(PAGE)

@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    report = str(data.get("report", ""))[:50000]
    if not report.strip():
        return jsonify(error="Please paste report text."), 400
    return jsonify(classify_report(report))

@app.post("/extract")
def extract():
    f = request.files.get("file")
    if not f:
        return jsonify(error="Choose a file first."), 400
    name = f.filename.lower()
    raw = f.read()
    try:
        if name.endswith(".txt"):
            text = raw.decode("utf-8", errors="replace")
        elif name.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                import io
                reader = PdfReader(io.BytesIO(raw))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return jsonify(error="For PDF text extraction, run: pip install pypdf"), 400
            if not text.strip():
                return jsonify(error="No selectable text found. This may be a scanned PDF; copy its text or configure OCR."), 400
        elif name.endswith((".png", ".jpg", ".jpeg")):
            try:
                from PIL import Image
                import pytesseract, io
                text = pytesseract.image_to_string(Image.open(io.BytesIO(raw)))
            except ImportError:
                return jsonify(error="For image OCR, install Pillow and pytesseract, plus the Tesseract OCR system program."), 400
            if not text.strip():
                return jsonify(error="No readable text found in the image. Try a clearer image or paste the report text."), 400
        else:
            return jsonify(error="Supported formats: TXT, PDF, PNG, JPG."), 400
        return jsonify(text=text[:50000])
    except Exception as e:
        return jsonify(error="Could not read this file. Try copying and pasting the report text instead."), 400


@app.get("/model/meta")
def model_meta():
    if not MODEL_READY:
        return jsonify(error="Optional ML packages are not installed: " + MODEL_ERROR), 503
    return jsonify(features=MODEL_FEATURES, metrics=MODEL_METRICS)

@app.get("/model/sample")
def model_sample():
    if not MODEL_READY:
        return jsonify(error="ML demo unavailable"), 503
    cls = 0 if request.args.get("cls") == "malignant" else 1
    idx = MODEL_RNG.choice(np.where(YTE == cls)[0])
    return jsonify(values=MODEL_XTE[idx].tolist(),
                   actual="malignant" if cls == 0 else "benign")

@app.post("/model/predict")
def model_predict():
    if not MODEL_READY:
        return jsonify(error="ML demo unavailable"), 503
    try:
        values = np.array((request.get_json(silent=True) or {})["values"], dtype=float).reshape(1, -1)
        if values.shape[1] != 30 or not np.isfinite(values).all():
            return jsonify(error="Exactly 30 finite numerical values are required."), 400
        probability = float(ENSEMBLE.predict_proba(values)[0, 0])
        return jsonify(estimate=probability,
                       note="Research-dataset demonstration only; not a patient's cancer probability.")
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Invalid measurement values."), 400


@app.get("/progress")
def progress_get():
    patient_id = request.args.get("patient_id", "").strip()
    records = load_records()
    if patient_id:
        records = [r for r in records if r.get("patient_id") == patient_id]
    return jsonify(records=records[-100:])

@app.post("/progress")
def progress_post():
    data = request.get_json(silent=True) or {}
    patient_id = re.sub(r"[^A-Za-z0-9_-]", "", str(data.get("patient_id", "")))[:40]
    date = str(data.get("date", ""))[:10]
    note = str(data.get("note", "")).strip()[:1000]
    if not patient_id or not note:
        return jsonify(message="Enter a patient code and a short note."), 400
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return jsonify(message="Please choose a valid visit date."), 400
    records = load_records()
    records.append({"id": uuid.uuid4().hex[:10], "patient_id": patient_id, "date": date, "note": note})
    save_records(records)
    return jsonify(message="Follow-up note saved on this computer.")

if __name__ == "__main__":
    print("Breast Health Report Companion")
    print("Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
