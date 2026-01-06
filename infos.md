Οκ, πάμε να το μαζέψουμε όλο από την αρχή μέχρι το τέλος, χωρίς να χαθεί τίποτα σημαντικό, αλλά και χωρίς να γίνει βιβλίο 😊

---

## 1. Αφετηρία

* Είχες ένα script τύπου `make_pseudo_noisy_plus.py` / `make_pseudo_noisy_plus_minpatch.py` που φτιάχνει **pseudo-noisy** εικόνες για να εκπαιδεύεις το **DnCNN** πάνω σε ESPI δεδομένα.
* Τα δεδομένα σου είναι οργανωμένα σε:

  * **Clean / averaged**: `C:\ESPI\data\wood_Averaged\w01, w02, w03` (και τελικά ειδικοί φάκελοι τύπου `W03_ESPI_90db-Averaged`)
  * **Real single-shot**: `C:\ESPI\data\wood_real_A\W01...`, `..._B\W02...`, `..._C\W03...`
  * ROI masks: `D:\ESPI_TEMP\roi_mask.png`, `roi_mask_W02.png`, `roi_mask_W03.png`
* Στόχος: να δημιουργήσεις **ρεαλιστικά noisy** για training/validation και να το τεκμηριώσεις για τη διπλωματική.

---

## 2. Πρώτο μεγάλο βήμα: pseudo-noisy για w01/w02/w03

* Έτρεξες το script πάνω στα clean averaged:

  * input: `C:\ESPI\data\wood_Averaged\...`
  * output: `C:\ESPI_DnCNN\pseudo_noisy\roi\w01|w02|w03`
  * με ROI μάσκες ανά board.
* Παρήχθησαν ~**699 pseudo-noisy** εικόνες (243 + 255 + 201).
* Τα pseudo-noisy ήταν **.tif**, τα averaged **.png** — το λάβαμε υπόψη στα manifests.

---

## 3. Manifests & LOBO

* Φτιάξαμε manifest για training/validation:

  * Train: w01 + w02
  * Val: w03
  * Αυτό ήταν το LOBO (Leave-One-Board-Out).
* Χρειάστηκε λίγο “debug” γιατί τα filenames δεν ταίριαζαν (π.χ. `_full.tif` vs `.png`), οπότε προσθέσαμε λογική **strip suffixes**.

---

## 4. Πρώτο DnCNN training (baseline)

* Έκανες ένα **pilot training** ( ~3 epochs) για να δεις αν δουλεύει το pipeline.
* Αποτελέσματα validation (w03):

  * PSNR γύρω στα **14–15 dB** μέσα στο training
  * αλλά όταν κάναμε inference πάνω στα πραγματικά averaged → **μόνο +0.0x dB** ή και χειρότερα.
* Συμπέρασμα τότε: *“το μοντέλο δεν βελτιώνει την ποιότητα”* → άρα υπάρχει **domain gap**.

---

## 5. Ablation: no-blur

* Έβγαλες νέα pseudo-noisy χωρίς blur (no-blur) και τα ξαναπέρασες.
* Ξανά training, ίδιο LOBO.
* Αποτέλεσμα: **χειρότερα** από το full noise (PSNR -0.3 dB) → blur component μάλλον ήταν χρήσιμο → άρα το πρόβλημα **δεν** ήταν το blur.

---

## 6. Hybrid dataset (70% full + 30% no-blur)

* Για να μειώσουμε το gap ανάμεσα σε train και val, φτιάξαμε **hybrid manifest**:

  * 70% από full pseudo-noisy
  * 30% από no-blur
* Το training με hybrid έδωσε **ελαφρώς καλύτερο** αποτέλεσμα:

  * PSNR +0.13 dB πάνω από το αρχικό baseline
  * SSIM +0.047
* Όμως ήταν ακόμα μικρό για να πεις “τέλεια”.

---

## 7. Log-domain training

* Κάναμε variant του DnCNN που δουλεύει σε **log-domain** (λογικό για speckle).
* Στο training φαινόταν “τρελά” καλά νούμερα (50–60 dB γιατί ήμασταν σε log scale), αλλά στο inference έπρεπε να κάνουμε **σωστό inverse** (`exp`, όχι `expm1`) και να συγκρίνουμε στο κανονικό domain.
* Μετά τη διόρθωση, το log-domain μοντέλο:

  * Ήταν το **καλύτερο από τα τρία**
  * Αλλά πάλι στο real validation έδινε **μικρά κέρδη** (+0.03 dB)
* Συμπέρασμα εκείνης της φάσης: το log-domain είναι η **σωστή κατεύθυνση**, αλλά **το real data μας “κρατάει πίσω”**.

---

## 8. Μεγάλο πρόβλημα: domain gap

* Όταν αξιολογήσαμε πάνω σε **πραγματικά** single → πραγματικά averaged (όχι pseudo-noisy), είδαμε:

  * PSNR gain: **αρνητικό** (π.χ. -0.26 dB)
  * SSIM gain: αρνητικό
  * MPI: ψηλό
* Άρα επιβεβαιώθηκε: **εκπαίδευση σε synthetic θόρυβο, test σε real → PSNR gap** (ό,τι λέει και η βιβλιογραφία).
* Κάναμε και δυο γύρους fine-tuning σε pseudo-real και πάλι είδαμε **-0.4 dB** και μετά **-0.8 dB** → δηλαδή το fine-tuning σε “όχι τέλειο” synthetic κάνει τα πράγματα χειρότερα.

---

## 9. “Pairfix” & σωστό alignment

Εκεί ήταν το turning point.

* Αντί να συγκρίνουμε “pseudo → pseudo-clean” ή “pseudo → averaged με άλλο όνομα”, κάναμε:

  * **real single** (π.χ. `C:\ESPI\data\wood_real_C\W03_ESPI_90db`)
  * ↔ **real averaged** (π.χ. `C:\ESPI\data\wood_Averaged\W03_ESPI_90db-Averaged`)
  * με **σωστό matching με βάση Hz & dB**
  * με **integer alignment** πριν το metric
  * με **strip suffixes**.
* Όταν το κάναμε έτσι (το ονομάσαμε ουσιαστικά “pairfix”), τα νούμερα εκτοξεύτηκαν:

  * **Mean ΔPSNR** πήγε από +0.028 dB → **+0.148 dB**
  * Με φιλτράρισμα 2 outliers → **+0.278 dB mean** και **+0.306 dB median**
  * **Success rate ~95.5%**
  * **MPI_norm ~0.09**
* Αυτό ήταν το πρώτο **“production-ready”** setup.

---

## 10. Long run (3 ώρες → 10 ώρες)

* Έτρεξες μεγάλο eval (≈3000 εικόνες).
* Στη μέση το σταμάτησες (ήταν αργό), αλλά οι **1119/2989** εικόνες που πρόλαβαν:

  * Median ΔPSNR ~ **+0.298 dB**
  * Mean ~ **+0.272 dB**
  * Success 95.5%
  * Outliers 0.2%
  * Σταθερή συμπεριφορά σε όλα τα Hz bands.
* Και αυτό το πακετάραμε σε **MASTER_SUMMARY_REPORT.md**.

---

## 11. Calibration v2.0

* Έκανες κανονικό calibration από:

  * single: `C:\ESPI\data\wood_real_C\W03_ESPI_90db`
  * avg:    `C:\ESPI\data\wood_Averaged\W03_ESPI_90db-Averaged`
* Τα πρώτα calibration είχαν θέματα:

  * R² ~0.69 (χαμηλό)
  * beta αρνητικό
  * με per-band είδαμε +0.19 dB (λίγο κάτω από στόχο)
  * με global medians χειρότερα
* Συμπέρασμα: **το working set σου (k=3.0, peak=60, sigma=0.01)** είναι πιο σταθερό από το αυτόματο calibration όταν το fit δεν είναι καλό.

---

## 12. Calibration v2.0 per board

* Το εφαρμόσαμε σε **W01** και **W02**.
* **W01**: πέτυχε τέλεια → **+0.307 dB** median ΔPSNR, ΔSSIM ~0.0054, MPI_norm <0.1.
* **W02**: απέτυχε και με τα δικά του params και με shared → το καταγράψαμε ως **limitation / board-specific mismatch**.

---

## 13. Τελική εικόνα (το “τελείωσε το pseudo-noisy” που λες)

Τελικός πίνακας που βγήκε:

| Board | Method                  | ΔPSNR         | Status            |
| ----- | ----------------------- | ------------- | ----------------- |
| W03   | working params (global) | +0.078 dB     | baseline, 100% ok |
| W01   | calib v2.0 per-band     | **+0.307 dB** | ✅ στόχος πιάστηκε |
| W02   | οποιαδήποτε μέθοδος     | < 0           | limitation        |

Άρα:

* έχεις **ένα board** (W01) που δείχνει ξεκάθαρα ότι το calibration δουλεύει
* έχεις **ένα board** (W03) που δείχνει σταθερό baseline
* έχεις **ένα board** (W02) που εξηγείς γιατί δεν δούλεψε → άρα η διπλωματική είναι πιο πιστευτή (δεν κρύβεις τα αρνητικά).

---

## 14. Τελικό συμπέρασμα (μια πρόταση)

> Όταν κάναμε αξιολόγηση πάνω σε λάθος ή ημι-συνθετικά ζεύγη, το DnCNN έδειχνε μηδενική ή αρνητική βελτίωση. Όταν όμως ταιριάξαμε **πραγματικά single-shot** με τα **αντίστοιχα averaged** (σωστό key, alignment, ROI) και χρησιμοποιήσαμε το log-domain μοντέλο, τότε πήραμε σταθερά **+0.27…+0.31 dB** σε ένα board (W01) και μικρό, αλλά θετικό κέρδος στο baseline (W03). Το W02 μένει ως documented limitation λόγω κακής βαθμονόμησης.

