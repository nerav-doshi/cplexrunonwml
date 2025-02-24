import sys, getopt

try:
    sys.modules['sklearn.externals.joblib'] = __import__('joblib')
    from ibm_watson_machine_learning import APIClient
except ImportError:
    from ibm_watson_machine_learning import APIClient


# THIS IS THE USER CREDENTIALS
wml_credentials = {
      "apikey": "jKKuPXaYb_Lz4Qd6jGDa8PBa4iwcvCOgPgHGjZ_3V29K",
      "url": "https://us-south.ml.cloud.ibm.com"
}


import base64
def getfileasdata(filename):
    with open(filename, 'r') as file:
        data = file.read();

    data = data.encode("UTF-8")
    data = base64.b64encode(data)
    data = data.decode("UTF-8")

    return data

def main(argv):
    cplex_file = "diet.lp"
    try:
        opts, args = getopt.getopt(argv,"hf:",["ffile="])
    except getopt.GetoptError:
        print('cplexrunonwml.py -f <file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('cplexrunonwml.py -f <file>')
            sys.exit()
        elif opt in ("-f", "--ffile"):
            cplex_file = arg
    print('CPLEX file is', cplex_file)

    basename = cplex_file.split('.')[0]
    model_name = basename + "_model"
    deployment_name = basename + "_deployment"
    space_name = basename + "_space"

    print("Creating WML Client")
    client = APIClient(wml_credentials)


    def guid_from_space_name(client, name):
        space = client.spaces.get_details()
        for item in space['resources']:
            if item['entity']["name"] == name:
                return item['metadata']['id']
        return None

    space_id = guid_from_space_name(client, space_name)

    if space_id == None:
        print("Creating space")
        cos_resource_crn = 'crn:v1:bluemix:public:cloud-object-storage:global:a/7f92ce1185a3460579ce2c76a03b1a67:69cd8af5-5427-4efd-9010-7ad13ac3e18a::'
        instance_crn = 'crn:v1:bluemix:public:pm-20:us-south:a/7f92ce1185a3460579ce2c76a03b1a67:82c6ef26-4fd2-40c4-95d3-abe3c3ad19fd::'

        metadata = {
            client.spaces.ConfigurationMetaNames.NAME: space_name,
            client.spaces.ConfigurationMetaNames.DESCRIPTION: space_name + ' description',
            client.spaces.ConfigurationMetaNames.STORAGE: {
                "type": "bmcos_object_storage",
                "resource_crn": cos_resource_crn
            },
            client.spaces.ConfigurationMetaNames.COMPUTE: {
                "name": "existing_instance_id",
                "crn": instance_crn
            }
        }
        space = client.spaces.store(meta_props=metadata)
        space_id = client.spaces.get_id(space)

    print("space_id:", space_id)

    client.set.default_space(space_id)

    print("Getting deployment")
    deployments = client.deployments.get_details()

    deployment_uid = None
    for res in deployments['resources']:
        if res['entity']['name'] == deployment_name:
            deployment_uid = res['metadata']['id']
            print("Found deployment", deployment_uid)
            break

    if deployment_uid == None:
        print("Creating model")
        import tarfile


        def reset(tarinfo):
            tarinfo.uid = tarinfo.gid = 0
            tarinfo.uname = tarinfo.gname = "root"
            return tarinfo


        tar = tarfile.open("model.tar.gz", "w:gz")
        tar.add(cplex_file, arcname=cplex_file, filter=reset)
        tar.close()

        print("Storing model")
        model_metadata = {
            client.repository.ModelMetaNames.NAME: model_name,
            client.repository.ModelMetaNames.DESCRIPTION: model_name,
            client.repository.ModelMetaNames.TYPE: "do-cplex_12.10",
            client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: client.software_specifications.get_uid_by_name(
                "do_12.10")
        }

        model_details = client.repository.store_model(model='./model.tar.gz', meta_props=model_metadata)

        model_uid = client.repository.get_model_uid(model_details)

        print(model_uid)

        print("Creating deployment")
        deployment_props = {
            client.deployments.ConfigurationMetaNames.NAME: deployment_name,
            client.deployments.ConfigurationMetaNames.DESCRIPTION: deployment_name,
            client.deployments.ConfigurationMetaNames.BATCH: {},
            client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {'name': 'S', 'nodes': 1}
        }

        deployment_details = client.deployments.create(model_uid, meta_props=deployment_props)

        deployment_uid = client.deployments.get_uid(deployment_details)

        print('deployment_id:', deployment_uid)

    print("Creating job")
    import pandas as pd

    solve_payload = {
        client.deployments.DecisionOptimizationMetaNames.SOLVE_PARAMETERS: {
            'oaas.logAttachmentName': 'log.txt',
            'oaas.logTailEnabled': 'true',
            'oaas.includeInputData': 'false',
            'oaas.resultsFormat': 'JSON'
        },
        client.deployments.DecisionOptimizationMetaNames.INPUT_DATA: [
            {
                "id": cplex_file,
                "content": getfileasdata(cplex_file)
            }
        ],
        client.deployments.DecisionOptimizationMetaNames.OUTPUT_DATA: [
            {
                "id": ".*\.json"
            },
            {
                "id": ".*\.txt"
            }
        ]
    }

    job_details = client.deployments.create_job(deployment_uid, solve_payload)
    job_uid = client.deployments.get_job_uid(job_details)

    print('job_id', job_uid)

    from time import sleep

    while job_details['entity']['decision_optimization']['status']['state'] not in ['completed', 'failed', 'canceled']:
        print(job_details['entity']['decision_optimization']['status']['state'] + '...')
        sleep(5)
        job_details = client.deployments.get_job_details(job_uid)

    print(job_details['entity']['decision_optimization']['status']['state'])

    for output_data in job_details['entity']['decision_optimization']['output_data']:
        if output_data['id'].endswith('csv'):
            print('Solution table:' + output_data['id'])
            solution = pd.DataFrame(output_data['values'],
                                    columns=output_data['fields'])
            solution.head()
        else:
            print(output_data['id'])
            if "values" in output_data:
                output = output_data['values'][0][0]
            else:
                if "content" in output_data:
                    output = output_data['content']
            output = output.encode("UTF-8")
            output = base64.b64decode(output)
            output = output.decode("UTF-8")
            print(output)
            with open(output_data['id'], 'wt') as file:
                file.write(output)

    # print ("Deleting deployment")
    # client.deployments.delete(deployment_uid)


if __name__ == '__main__':
    main(sys.argv[1:])
