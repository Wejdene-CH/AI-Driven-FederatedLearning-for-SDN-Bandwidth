#!/usr/bin/env python3

from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import logging
import subprocess

# Configuration des journaux
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Switch personnalisé pour activer sFlow
class SFlowSwitch(OVSSwitch):
    def start(self, controllers):
        super().start(controllers)
        self.cmd(f'ovs-vsctl -- set bridge {self.name} sflow=@sFlow -- ' +
                 '--id=@sFlow create sflow target="127.0.0.1:6343" ' +
                 'sampling=10 polling=10')
        logging.info(f'sFlow activé sur le switch {self.name}')

# Switch personnalisé pour capturer les paquets avec tcpdump
class LoggingSwitch(OVSSwitch):
    def start(self, controllers):
        super().start(controllers)
        self.capture = subprocess.Popen([
            'tcpdump', 
            '-i', self.name + '-eth1', 
            '-w', f'/tmp/{self.name}_packets.pcap',
            '-n'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f'Tcpdump activé sur le switch {self.name}, capture sauvegardée dans /tmp/{self.name}_packets.pcap')

    def stop(self):
        super().stop()
        if hasattr(self, 'capture'):
            self.capture.terminate()
            logging.info(f'Tcpdump arrêté pour {self.name}')

# Fonction pour créer et surveiller la topologie
def setup_monitored_network():
    logging.info("Création de la topologie...")
    net = Mininet(controller=RemoteController, switch=SFlowSwitch)

    # Ajout d'un contrôleur
    controller = net.addController('c0', ip='127.0.0.1', port=6653)
    logging.info("Contrôleur ajouté : c0")

    # Ajout des commutateurs et hôtes
    s1 = net.addSwitch('s1', cls=LoggingSwitch)
    h1 = net.addHost('h1', ip='10.0.0.1', mac='00:00:00:00:00:01')
    h2 = net.addHost('h2', ip='10.0.0.2', mac='00:00:00:00:00:02')

    logging.info("Commutateur s1 et hôtes h1, h2 ajoutés")

    # Création des liens
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    logging.info("Liens entre les hôtes et le commutateur créés")

    # Démarrage du réseau
    net.start()
    logging.info("Topologie démarrée")

    return net

if __name__ == "__main__":
    setLogLevel('info')

    # Initialisation du réseau
    net = setup_monitored_network()

    try:
        # Tests basiques
        logging.info("Exécution d'un test ping entre h1 et h2...")
        h1, h2 = net.get('h1', 'h2')
        result = h1.cmd('ping -c 3', h2.IP())
        logging.info(f"Résultats du test ping :\n{result}")
        
        # Lancement de l'interface CLI pour exploration
        CLI(net)

    finally:
        # Arrêt du réseau
        logging.info("Arrêt du réseau...")
        net.stop()
        logging.info("Réseau arrêté avec succès")
