import os
import glob
import logging

logger = logging.getLogger(__name__)


def get_ascend_devices():
    devices = []
    pci_path = '/sys/bus/pci/devices/*'
    
    for dev in glob.glob(pci_path):
        try:
            vendor_path = os.path.join(dev, 'vendor')
            device_path = os.path.join(dev, 'device')
            
            if os.path.exists(vendor_path):
                with open(vendor_path, 'r') as f:
                    vendor = f.read().strip()
                
                if vendor == "0x19e5" and os.path.exists(device_path):
                    with open(device_path, 'r') as f:
                        device = f.read().strip()
                        devices.append(device)
        except (IOError, OSError) as e:
            logger.warning(f"can not fetch device {dev}: {e}")
            continue
    
    if not devices:
        print("no device info read in /sys/bus/pci/devices, is_compile_on_910_95 set to False by default") 
    return devices


ascend_devices = get_ascend_devices()
is_compile_on_910_95 = any("0xd806" in dev for dev in ascend_devices)