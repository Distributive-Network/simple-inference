/**
 * @file   inference.js
 *         Inference using some machine model via DCP
 *
 * @author Ryan Saweczko, ryansaweczko@distributive.network
 */
'use strict';

const path = require('path');
const fs = require('fs');
const { workFunction } = require('./workFunction');

/**
 * Deploy work function for model inferencing via DCP
 */
async function deploy(inputSet, model, computeGroup, output, webgpu)
{
	const compute = require('dcp/compute');

  const labels = { modelName: model.name, projectID: Date.now(), debug: false, webgpu };
  const args = [labels];
  if (model.modelDownload)
  {
    const url = new URL(model.modelDownload);
    args.push(new URL(`/${model.preprocess}`, url));
    args.push(new URL(`/${model.postprocess}`, url));
    args.push(model.packages);
    args.push(new URL(`/${model.model}`, url));
  }
	let job = compute.for(inputSet, workFunction, args);

  job.public.name = `DCP Inferencing: ${model.name}`;
  job.requires('onnxruntime-dcp/dcp-wasm.js');
  job.requires('onnxruntime-dcp/dcp-ort.js');
  job.requires('pyodide-core/pyodide-core.js');
  if (!model.modelDownload)
    job.requires(`${model.name}/module.js`);

  if (computeGroup)
    job.computeGroups    = [computeGroup];
	job.collateResults   = false;
	job.workerConsole    = true;
  job.requirements.environment = { webgpu, };

	job.on('accepted', async () => {
		console.log(`Job has dcp id: ${job.id} and has been accepted...`);
	});

	job.on('result', (result) => {
    console.log(result.result);
  });

	job.on('cancel', async (error) => {
    console.log('Job cancelled', error);
	});

	job.on('error', async function errorHandler(err) {
		console.error(err);
	});
  job.on('console', (event) => {
    if (event.message[0].common) {
      console.log('ONNX Runtime Version:', event.message[0].common);
    }
  });

  let resultSet = [];
  try {
    resultSet = await job.exec();
  } catch(err) {
    throw new Error(`Failed to execute job: ${err.message}`);
  }
  
  if (output)
    fs.writeFileSync(output, JSON.stringify(Array.from(resultSet)));

  return resultSet;
}

function usage()
{
  console.log(`inference.js: Perform inferencing on upload a machine learning model via DCP.
Usage:
    node inference.js </path/to/model.json> </path/to/input/dir>
                      [--batch=<size>] [--output=<outputFile>]
                      [--computeGroup=<joinKey,joinSecret>] [--webgpu] [--help]

Where:
    --batch         is the batch size for each slice
    --output        is the output file (json output)
    --computeGroup  is the compute group to deploy the job into
    --webgpu        enables webgpu as the execution provider
    --help          output help menu and exit
`);
  process.exit(1);
}

exports.deploy = deploy;

if (require.main === module)
{
  if (process.argv.length < 4)
    usage();

  var batchSize = 1;
  var outputFile, computeGroup, webgpu;
  // parse cli options
  for (let i = 4; i < process.argv.length; i++)
  {
    const arg = process.argv[i]; 
    if (arg.startsWith('--help'))
      usage();
    else if (arg.startsWith('--batch'))
    {
      let [, size] = arg.split('=');
      if (!size)
      {
        size = process.argv[i+1];
        i++;
      }
      batchSize = parseInt(size);
    }
    else if (arg.startsWith('--output'))
    {
      let [, output] = arg.split('=');
      if (!output)
      {
        output = process.argv[i+1];
        i++;
      }
      outputFile = output;
    }
    else if (arg.startsWith('--computeGroup'))
    {
      let [, cg] = arg.split('=');
      if (!cg)
      {
        cg = process.argv[i+1];
        i++;
      }
      const [joinKey, joinSecret] = cg.split(',');
      computeGroup = { joinKey, joinSecret };
    }
    else if (arg.startsWith('--webgpu'))
    {
      webgpu = true;
    }
  }

  const modelInfo = JSON.parse(fs.readFileSync(process.argv[2], { encoding: 'utf-8' }));
  const inputDir = process.argv[3]
  const inputFilenames = fs.readdirSync(inputDir)
  const inputSet = [];
  let index = 0;
  for (let file of inputFilenames)
  {
    if (index === 0)
      inputSet.push({b64Data: {}})
    inputSet[inputSet.length - 1]['b64Data'][file] = fs.readFileSync(inputDir + '/' + file).toString('base64');
    index++;
    index %= batchSize
  }

  require('dcp-client').init().then(() => {
    deploy(inputSet, modelInfo, computeGroup, outputFile, webgpu);
  });
}


