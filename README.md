# AIC-adversarial-image-attack-lab

A hands-on lab to explore adversarial attacks on image classifiers (FGSM, PGD, DeepFool) and simple defenses.
Includes a Streamlit UI, training notebooks and Docker support.

## Quickstart (Docker)
Build:
  docker build -t aic-adv-lab .

Run:
  docker run -p 8501:8501 aic-adv-lab

Open: http://localhost:8501

## Project layout
See folders: data/, models/, attacks/, defenses/, app/, notebooks/, utils/

## License
MIT
