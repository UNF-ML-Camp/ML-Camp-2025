# Velkommen til UNFs tegn og gæt konkurrence
Kender du til spillet [Quick, Draw!](https://quickdraw.withgoogle.com/)? I dette modul skal du træne din/jeres egen algoritme til at genkende tegninger ("doodles") lige som den algoritme du kan se i spillet. De algoritmer vi kommer til at bygge er neurale netværk, ligesom dem i lige har lært om.

## Introduktion
Jeres opgave er:
- At definere model architecturen
- Finde de rigtige hyperparametre
- Træne modellen med forskellige kombinationer af model architecturer og hyperparametre

Når I har fundet en model i er glad for, kan i dele den med det faglige team (nærmere info følger om hvordan) og til sidst afholder vi en tegn og gæt konkurrence med alle modellerne for at se hvem der har bygget den bedste tegn og gæt model.

Herunder følger en dokumentation på koden basen du skal bruge under konkurrencen

## Dokumentation
Kodebasen består af følgende filer:
```python
QuickDraw     
├── data                    # Folder til at gemme en lokal version af de data filer vi bruger til træning
│   └── _data_static.py     # Giver labels på de klasser vi træner på (Niks pille ;) )
├── saved_models            # Folder hvor jeres gemte modeller i har trænet ligger i
├── saved_models            # Folder hvor jeres gemte test session (som .csv-filer) ligger i 
├── analyze_data.ipynb      # ~ Notebook til analysering af datasættet og dataloaderne ~
├── app.py                  # Application i kan bruge til at teste gemte modeller lokalt
├── get_data.py             # Definerer jeres data loader til træning
├── main.py                 # ~ Her skal du vælge hyperparametre og træne modellen fra ~
├── model.py                # ~ Her skal I definere model arkitekturen ~
├── options.py              # ~ Her kan O tilføje nye hyperparametre, hvis I vil bruge noget mere eksotisk ~
├── train.py                # Træningsproceduren (Niks pille ;) )
└── README.md               # Denne fil
```
Det er generalt set en god ide at læse kodebasen igennem før i går igang med at bygge og træne modeller. Hvis i har spørgsmål kan i altid tage fat i faglig for at få hjælp og forklaringer.

## Arbejdsgang
I kommer mest af alt til at arbejde fra 3 filer: `main.py`, `model.py` og `options.py`. Her følger lidt mere info om hvordan i kan bruge kodebasen til træning og evaluering af modeller.

### 1. Naviger til QuickDraw mappen
Åben jeres terminal og skriv
```bash
cd 2.NN/QuickDraw/
```

### 2. Analyser datasættet
Naviger til Notebooken `analyze_data.ipynb`. Her har vi forberedt lidt dataanalyser, der giver nogle ideer om hvordan vores data er fordelt, hvordan vores gennemsnitlige doodle for et bestemt label ser ud, og hvordan klasserne loades i batches og epochs. Overvej at tage fat i en faglig og tal analyserne igennem med dem.

Generelt kan data-analyserne bruges til at undersøge hvilke data punkter der muligvis kan være svære for det neurale netværk at lære. Dette kan informere hvad vi især ønsker at teste vores senere trænede modeller op imod i vores live-testing.

### 3. Definer modellen
I kan definere jeres model architektur i `model.py` filen. Overvej de ting vi har lært de seneste par dage, når i tager arkitektur valg. I kan også altid overveje at snakke med en faglig :sunglasses:

### 4. Vælg hyperparametre
Default hyperparametre er givet i `options.py`, men en default værdi. Afhængig af hvilken optimizer i gerne vil bruge, kan der være andre hyperparametre der også giver mening for jer at inkludere i klassen.

I kan vælge andre værdier til en bestemt hyper parameter i `main.py`. Læg mærke til at default værdierne ikke er optimale, så hvis i gerne vil vinde konkurrencen er det nok en god ide at optimere hyperparametrene her.

# 5. Træn modellen
I træne den ved at følgende kommando i terminalen:
```bash
python main.py
```

Den kommando kører alt i `main.py`filen, der gør følgende:
1. Henter dataen
2. Definerer hyperparametre (efter de værdier i har indsat i `main.py`)
3. Definerer modellen (efter den arkitektur i har bygget i `model.py`)
4. Træner modellen (og logger den med `mlflow`)
5. Gemmer modellen i `saved_models`

### 6. Sammenlign modeller
Eftersom vi logger modellerne i `mlflow` under træning, kan vi når vi har bygget flere modeller sammenligne med visuel:
```bash
mlflow server
```
Hvis i derfra navigerer til `http://127.0.0.1:5000` (jeres localhost) får i et dashboard i kan bruge til at sammenligne forskellige modeller. Læg mærke til at i kun kan have dashboardet åbent i én fane.

### 7. Test jeres modeller
Hvis I gerne vil teste jeres trænede modeller med ny test data (som i selv generer) , kan i teste dem i med live tegninger i `app.py`:
```bash
streamlit run app.py
```

I frontenden har i mulighed for at opsamle modellernes prædiktioner og performance, samt at gemme denne i mappen `test_session` som .csv filer. I kan lave yderligere analyse af modellerne med disse filer for at beslutte jer for hvilken fil i indsender til konkurrencen.

# :sparkles: Vi ønsker held og lykke med konkurrencen :sparkles:




