# Segmentazione batch di polmoni e vasi con haversack

Fork personale di [mhalle/haversack](https://github.com/mhalle/haversack) con uno script per
segmentare in batch una cartella di TC toraciche: lobi polmonari, trachea, vie aeree, arterie e
vene polmonari, con le mappe di probabilità del modello.

- **Autore delle modifiche:** Edoardo De Rose
- **Repository:** https://github.com/edoardoedr/haversack
- **Progetto originale:** [haversack](https://github.com/mhalle/haversack) (Apache 2.0). La guida
  completa del pacchetto è in [README.md](README.md).

## Cosa aggiunge questo fork

| File | Cosa fa |
|---|---|
| [batch_segment.py](batch_segment.py) | segmenta ogni paziente di una cartella con i task indicati nella config |
| [configs/lungs.toml](configs/lungs.toml) | config di esempio: lobi e trachea, vasi e vie aeree con probabilità |

## Installazione

Su un server Linux con GPU NVIDIA, in un ambiente conda:

```bash
conda create -n haversack python=3.12 -y
conda activate haversack
pip install uv                    # uv solo come installer, dentro l'ambiente conda
git clone https://github.com/edoardoedr/haversack.git && cd haversack
uv pip install -e ".[cuda]"       # installa nell'ambiente conda attivo
python batch_segment.py configs/lungs.toml
```

Note:

- **Perché uv e non pip.** Alcune dipendenze di haversack (`duckn`, `rankfield`, `provender`,
  `fastsurfer-lean`) non sono su PyPI: si installano da tag git dichiarati in
  `[tool.uv.sources]` del `pyproject.toml`, che solo uv legge. `uv pip install` installa
  nell'ambiente conda attivo e non crea un `.venv` separato.
- **`[cuda]`** aggiunge il backend Triton per le GPU NVIDIA. Senza GPU NVIDIA si installa con
  `uv pip install -e .`
- **Pesi dei modelli.** Al primo utilizzo vengono scaricati da internet (qualche centinaio di MB
  per modello) in `~/.totalsegmentator/nnunet/results`. Se il server non ha accesso a internet,
  copiali lì a mano oppure imposta `TOTALSEG_WEIGHTS_PATH`.
- **Controllo dell'installazione:** `haversack tasks ts.v2:lung_vessels` deve elencare le 4
  strutture del task.

### Driver NVIDIA più vecchio di CUDA 13

Da PyPI arriva torch compilato per CUDA 13 (`2.14.1+cu130`), che richiede un driver NVIDIA 580
o più recente. Con un driver più vecchio (`nvidia-smi` mostra per esempio `Driver Version:
535.288.01  CUDA Version: 12.4`) ogni segmentazione fallisce con:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12040)
```

Se non puoi aggiornare il driver, reinstalla torch e torchvision compilati per CUDA 12.6, che
funzionano con i driver della serie 12. Vanno reinstallati **insieme**: se resta il torchvision
di PyPI, l'errore diventa `operator torchvision::nms does not exist`.

```bash
uv pip install --reinstall "torch==2.14.1" "torchvision==0.29.1" --index-url https://download.pytorch.org/whl/cu126
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available()); from torchvision.ops import nms; print('ok')"
```

Deve stampare `2.14.1+cu126 0.29.1+cu126 True` e poi `ok`. Verificato su un server con driver
535.288.01 e una A100 (partizione MIG 2g.20gb), il 2026-10-24. Se la versione di torch installata
da haversack cambia, adatta i due numeri: torchvision 0.29 corrisponde a torch 2.14.

## Uso

1. Metti le TC in una cartella, un file `.nii.gz` o `.nii` per paziente. Il nome del file,
   senza estensione, diventa il nome del paziente.
2. In [configs/lungs.toml](configs/lungs.toml) imposta `input_dir` e `output_dir`.
3. Lancia:

   ```bash
   conda activate haversack
   python batch_segment.py configs/lungs.toml
   ```

Lo script salta i pazienti già completati, quindi dopo un'interruzione basta rilanciarlo.
Se un paziente fallisce, lo segnala e prosegue con gli altri.

Tempi misurati su una A100 (partizione MIG 2g.20gb) con un paziente (PD-1-Lung-00043): lobi e trachea 21 s, vasi e vie aeree con store e mappe di probabilità 75 s. In
quella prova la classe più probabile delle mappe differiva dalle maschere nello 0.0008 % dei voxel.

## La config

```toml
input_dir = "/path/to/patients"          # ogni *.nii.gz / *.nii è un paziente
output_dir = "/path/to/segmentations"    # una sottocartella per paziente

device = "auto"            # auto, mps, cuda, cpu
overwrite = false          # true rifà anche i pazienti già completati
save_multilabel = false    # true salva anche la label map completa di ogni task
cache_models = 1           # modelli tenuti in memoria tra un paziente e l'altro

[[segmentations]]
task = "ts.v2:total"       # i modelli a 1.5 mm, la risoluzione più fine di TotalSegmentator
roi = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
       "lung_middle_lobe_right", "lung_lower_lobe_right", "trachea"]
roi_subset = true          # esegue solo il modello che contiene queste strutture

[[segmentations]]
task = "ts.v2:lung_vessels"
save_logits = true         # salva i logits come ranked store (lung_vessels.duckn.zip)
probabilities = "float32"  # mappe di probabilità dallo store: "float32" o "uint8"
```

Per ogni voce `[[segmentations]]`:

| Chiave | Significato |
|---|---|
| `task` | il task di haversack (`haversack tasks` li elenca tutti) |
| `roi` | le strutture da salvare; senza `roi` le salva tutte |
| `roi_subset` | con `roi` impostato, esegue solo i modelli che contengono quelle strutture (default `true`) |
| `save_logits` | salva anche il ranked store `<task>.duckn.zip` |
| `probabilities` | `"float32"` o `"uint8"`: mappe di probabilità ricavate dallo store (richiede `save_logits`) |

## Output

Per ogni paziente, in `output_dir/<paziente>/`:

| File | Contenuto |
|---|---|
| `<struttura>.nii.gz` | maschera binaria (0/1) sulla griglia e nell'orientamento della TC |
| `lung_vessels.duckn.zip` | ranked store: i logits del modello in forma compressa, sulla griglia del modello |
| `<struttura>_prob.nii.gz` | probabilità della struttura sulla griglia della TC: 0–1 in float32, 0–255 in uint8 (dividere per 255) |

Strutture prodotte con la config di esempio:

- da `ts.v2:total`: `lung_upper_lobe_left`, `lung_lower_lobe_left`, `lung_upper_lobe_right`,
  `lung_middle_lobe_right`, `lung_lower_lobe_right`, `trachea`;
- da `ts.v2:lung_vessels`: `lung_airways`, `lung_airways_wall`, `lung_arteries`, `lung_veins`.

## Dettagli di metodo

**Risoluzione.** I lobi e la trachea vengono da `ts.v2:total`, i modelli a 1.5 mm di
TotalSegmentator; `total_fast` è quello a 3 mm. Le maschere vengono riportate sulla griglia della
TC originale.

**roi_subset.** `ts.v2:total` è composto da cinque modelli (organi, vertebre, cuore, muscoli,
coste). Le sei strutture polmonari stanno tutte nel modello degli organi (Dataset291), quindi lo
script esegue solo quello: circa 5 volte meno calcolo. In `total` le parti successive
sovrascrivono le precedenti dove si sovrappongono, mentre qui il modello degli organi resta solo:
ai bordi, per esempio tra lobo e cuore, possono cambiare pochi voxel. È lo stesso comportamento di
`--roi_subset` in TotalSegmentator. Con `roi_subset = false` si ottiene il risultato identico a
`total`.

**Ranked store.** haversack non salva i logits grezzi. Per ogni voxel conserva le classi in ordine
di probabilità e di quanti logit ciascuna è dietro la vincitrice, quantizzati a 8 bit. Ne derivano
alcune proprietà:

- le probabilità si ricostruiscono con un errore massimo di circa 0.004;
- una classe più di 8 logit dietro la vincitrice è salvata come assente (probabilità sotto circa
  3e-4);
- per `lung_vessels`, con 5 classi, vengono conservate tutte;
- del primo stadio della cascata, il `total_fast` usato per ritagliare l'area dei polmoni, non si
  salvano i logits.

Lo store si rilegge con haversack (`haversack restore`, `haversack.ranked`).

**Mappe di probabilità.** Si ricavano dallo store: i deficit di ogni classe
(`logit_classe − logit_vincente`) vengono interpolati linearmente sulla griglia della TC, poi si
applica la softmax voxel per voxel. Dato che la softmax non cambia se si sottrae a tutti i logit
la stessa costante, il risultato equivale a interpolare i logits veri: è lo stesso campo da cui
haversack ricava le maschere. Per ogni paziente lo script stampa per quale percentuale di voxel la
classe più probabile differisce dalle maschere: deve essere molto piccola. Fuori dal ritaglio della
cascata il modello non ha girato, e lì lo sfondo vale 1.

Se si aggiunge `probabilities` a una config già eseguita, al rilancio le mappe vengono ricavate
dagli store esistenti, senza rifar girare la rete.

## Aggiornare dal progetto originale

```bash
git remote add upstream https://github.com/mhalle/haversack.git   # una volta sola
git fetch upstream
git merge upstream/main
```

## Licenza e citazioni

haversack è distribuito sotto licenza Apache 2.0 (vedi [LICENSE](LICENSE)). Le modifiche di
questo fork sono i file elencati in "Cosa aggiunge questo fork". I modelli di TotalSegmentator
vanno citati secondo le indicazioni dei loro autori: `haversack cite ts.v2:total` e
`haversack cite ts.v2:lung_vessels` stampano licenza e riferimenti.
