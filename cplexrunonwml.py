import sys, getopt

try:
    sys.modules['sklearn.externals.joblib'] = __import__('joblib')
    from watson_machine_learning_client import WatsonMachineLearningAPIClient
except ImportError:
    from watson_machine_learning_client import WatsonMachineLearningAPIClient


# THIS IS THE USER CREDENTIALS
wml_credentials = {
    "apikey": "xxxxxxxxxxxxxxxxxxxxxxxxx",
    "instance_id": "xxxxxxxxxxxxxxxxxxxxx",
    "url": "https://us-south.ml.cloud.ibm.com",
}
# END OF THE USER CREDENTIALS

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

    print("Creating WML Client")
    client = WatsonMachineLearningAPIClient(wml_credentials)

    print("Getting deployment")
    deployments = client.deployments.get_details()

    deployment_uid = None
    for res in deployments['resources']:
        if res['entity']['name'] == deployment_name:
            deployment_uid = res['metadata']['guid']
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
            client.repository.ModelMetaNames.RUNTIME_UID: "do_12.10"
        }

        model_details = client.repository.store_model(model='./model.tar.gz', meta_props=model_metadata)

        model_uid = client.repository.get_model_uid(model_details)

        print(model_uid)

        print("Creating deployment")
        deployment_props = {
            client.deployments.ConfigurationMetaNames.NAME: deployment_name,
            client.deployments.ConfigurationMetaNames.DESCRIPTION: deployment_name,
            client.deployments.ConfigurationMetaNames.BATCH: {},
            client.deployments.ConfigurationMetaNames.COMPUTE: {'name': 'S', 'nodes': 1}
        }

        deployment_details = client.deployments.create(model_uid, meta_props=deployment_props)

        deployment_uid = client.deployments.get_uid(deployment_details)

        print('deployment_id:', deployment_uid)

    print("Creating job")
    import pandas as pd

    with open(cplex_file, 'r') as file:
        model = file.read();
    import base64

    model = model.encode("UTF-8")
    model = base64.b64encode(model)
    model = model.decode("UTF-8")
    df_model = pd.DataFrame(columns=['___TEXT___'], data=[[model]])
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
                "values": df_model
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
            output = output_data['values'][0][0]
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
