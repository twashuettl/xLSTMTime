# update system
apt-get update
apt-get upgrade -y
# install Linux tools and Python 3
apt-get install software-properties-common wget curl \
    python3-dev python3-pip python3-wheel python3-setuptools git -y
# install Python packages
rm /usr/lib/python3.12/EXTERNALLY-MANAGED
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
pip3 cache purge
apt-get autoremove -y
apt-get clean