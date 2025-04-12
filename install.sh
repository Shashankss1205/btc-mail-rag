mkdir -p ./data/
cd ./data/
git clone --mirror https://gnusha.org/pi/bitcoindev
ls -la bitcoindev.git
mkdir -p bitcoin_resources && git clone bitcoindev.git bitcoin_resources
if python --version 2>&1 | grep -q "Python 3"; then
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    python3 main.py
else
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python main.py
fi