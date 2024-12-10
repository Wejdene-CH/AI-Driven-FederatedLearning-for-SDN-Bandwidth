#sudo python3 sdn_topology.py
#!/usr/bin/env python3

from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel

def create_topology():
    # Create Mininet with RemoteController
    net = Mininet(controller=RemoteController)
    
    # Add controller
    net.addController('c0', ip='127.0.0.1', port=6653)
    
    # Add switches
    s1 = net.addSwitch('s1')
    
    # Add hosts with MAC and IP
    h1 = net.addHost('h1', mac='00:00:00:00:00:01', ip='10.0.0.1/24')
    h2 = net.addHost('h2', mac='00:00:00:00:00:02', ip='10.0.0.2/24')
    
    # Create links
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    
    # Start the network
    net.start()
    
    # Open CLI for interaction
    CLI(net)
    
    # Stop the network
    net.stop()

# Main execution
if __name__ == '__main__':
    # Set logging level
    setLogLevel('info')
    
    # Create the topology
    create_topology()