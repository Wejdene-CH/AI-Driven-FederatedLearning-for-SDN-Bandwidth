#!/usr/bin/env python3

from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import logging
import subprocess
from datetime import datetime

# Configuration des journaux
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Switch personnalisé pour capturer les paquets avec tcpdump
class LoggingSwitch(OVSSwitch):
    def start(self, controllers):
        super().start(controllers)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.capture = subprocess.Popen([
            'tcpdump', 
            '-i', self.name + '-eth1', 
            '-w', f'/tmp/{self.name}_packets_{timestamp}.pcap',
            '-n','-vv'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f'Tcpdump activé sur le switch {self.name}, capture sauvegardée dans /tmp/{self.name}_packets_{timestamp}.pcap')

    def stop(self):
        super().stop()
        if hasattr(self, 'capture'):
            self.capture.terminate()
            logging.info(f'Tcpdump arrêté pour {self.name}')

# Fonction pour créer et surveiller la topologie
def setup_monitored_network():
    logging.info("Création de la topologie...")

    # Création du réseau
    net = Mininet(controller=RemoteController, switch=LoggingSwitch)

    # Ajout d'un contrôleur
    controller = net.addController('c0', ip='127.0.0.1', port=6653)
    logging.info("Contrôleur ajouté : c0")

    # Ajout des commutateurs
    main_switch = net.addSwitch('s1')
    leaf_switch1 = net.addSwitch('s2')
    leaf_switch2 = net.addSwitch('s3')
    logging.info("Commutateurs s1, s2 et s3 ajoutés")

    # Ajout des hôtes pour le premier switch
    h1 = net.addHost('h1', ip='10.0.0.1', mac='00:00:00:00:00:01')
    h2 = net.addHost('h2', ip='10.0.0.2', mac='00:00:00:00:00:02')
    net.addLink(h1, leaf_switch1)
    net.addLink(h2, leaf_switch1)
    logging.info("Hôtes h1 et h2 connectés à s2")

    # Ajout des hôtes pour le deuxième switch
    h3 = net.addHost('h3', ip='10.0.1.1', mac='00:00:00:00:00:03')
    h4 = net.addHost('h4', ip='10.0.1.2', mac='00:00:00:00:00:04')
    net.addLink(h3, leaf_switch2)
    net.addLink(h4, leaf_switch2)
    logging.info("Hôtes h3 et h4 connectés à s3")

    # Création des liens entre switches
    net.addLink(main_switch, leaf_switch1)
    net.addLink(main_switch, leaf_switch2)
    logging.info("Liens entre s1 et s2, s1 et s3 créés")

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
        logging.info("Exécution d'un test ping...")
        net.pingAll()
        
        # Lancement de l'interface CLI pour exploration
        CLI(net)

    finally:
        # Arrêt du réseau
        logging.info("Arrêt du réseau...")
        net.stop()
        logging.info("Réseau arrêté avec succès")
