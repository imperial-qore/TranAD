import subprocess
from time import sleep
import json

vmlist = [('vm1', 'Standard_B2ms'), ('vm2', 'Standard_B2ms')]

HEADER = '\033[1m'
FAIL = '\033[91m'
ENDC = '\033[0m'

def run(cmd, shell=True):
  data = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if 'ERROR' in data.stderr.decode():
    print(cmd)
    print(FAIL)
    print(data.stderr.decode())
    print(ENDC)
    exit()
  return data.stdout.decode()

#################

print(f'{HEADER}Create Azure VM{ENDC}')
for name, size in vmlist:
  run(f'az vm create --resource-group vm1_group --name {name} --size {size} --image UbuntuLTS --ssh-key-values id_rsa.pub --admin-username ansible')

# #################

print(f'{HEADER}Wait for deployment (1 minute){ENDC}')
sleep(60)

#################

print(f'{HEADER}Open port 8081{ENDC}')
for name, size in vmlist:
  run(f'az vm open-port --resource-group vm1_group --name {name} --port 8081')

#################

print(f'{HEADER}Install new kernel{ENDC}')
for name, size in vmlist:
  cmd = f"az vm run-command invoke -g vm1_group -n {name} --command-id RunShellScript --scripts 'sudo apt install -y -f linux-image-4.15.0-1009-azure linux-tools-4.15.0-1009-azure linux-cloud-tools-4.15.0-1009-azure linux-headers-4.15.0-1009-azure linux-modules-4.15.0-1009-azure linux-modules-extra-4.15.0-1009-azure'"
  run(cmd)

#################

print(f'{HEADER}Deallocate VMs{ENDC}')
for name, size in vmlist:
  run(f'az vm deallocate --resource-group vm1_group --name {name}')

#################

print(f'{HEADER}Update Disks{ENDC}')
data = run("az disk list --resource-group vm1_group")
d = eval(data.replace('\r\n', '').replace('null', "'null'"))
for a in d:
  disk_id = a['id']
  disk_id = disk_id.split('/')[-1]
  run(f'az disk update --name {disk_id} --resource-group vm1_group --size-gb 256 --set tier=P15')

#################

print(f'{HEADER}Start VMs{ENDC}')
for name, size in vmlist:
  run(f'az vm start --resource-group vm1_group --name {name}')

#################

print(f'{HEADER}Create storage account')
run(f'az storage account create --name shreshthstorage --resource-group vm1_group --access-tier Hot --kind StorageV2 --location uksouth --sku Standard_RAGRS')

#################

print(f'{HEADER}Set diagnostic info')
sas_token = run(f'az storage account generate-sas --account-name shreshthstorage --expiry 2037-12-31T23:59:00Z --permissions wlacu --resource-types co --services bt -o tsv')
sas_token = sas_token.strip()
p_settings = '{'+f"'storageAccountName': 'shreshthstorage', 'storageAccountSasToken': '{sas_token}'"+'}'
with open('diagnostic.json') as f:
  setting = json.load(f)
for name, size in vmlist:
  setting['ladCfg']['resourceId'] = name
  sett = json.dumps(setting).replace('"', "'")
  cmd = f'az vm diagnostics set --settings "{sett}" --protected-settings "{p_settings}" --resource-group vm1_group --vm-name {name}'
  run(cmd)
