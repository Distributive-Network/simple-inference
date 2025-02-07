#!/usr/bin/env node
/**
 * @file   upload-model.js
 *         Upload a model based on a model descriptor json file
 *
 * @author Ryan Saweczko, ryansaweczko@distributive.network
 */
'use strict';

const fs = require('fs');
const path = require('path');

function usage()
{
  console.log(`upload-model: upload a machine learning model to use with DCP.
Usage:
    node upload-model.js /path/to/model.json`);
  process.exit(1);
}

if (process.argv.length !== 3)
  usage();

const modelInfo = JSON.parse(fs.readFileSync(process.argv[2], { encoding: 'utf-8' }));
const dirname = path.dirname(process.argv[2]);

const model = fs.readFileSync(dirname + '/' + modelInfo.model).toString('base64');
const preprocess = fs.readFileSync(dirname + '/' + modelInfo.preprocess).toString('base64');
const postprocess = fs.readFileSync(dirname + '/' + modelInfo.postprocess).toString('base64');

const bravojsModule = `
module.declare([], async function bootstrap_decl (require, exports, module) {
  exports.model = "${model}";
  exports.preprocess = "${preprocess}";
  exports.postprocess = "${postprocess}";
  exports.packages = ${JSON.stringify(modelInfo.packages)};
});
`
const packageText = `{
  "name": "${modelInfo.name}",
  "version": "${modelInfo.version}",
  "files": {
    "module.js": "module.js"
  }
}
`

const dir = `upload-${Date.now()}`

fs.mkdirSync(dir);
fs.writeFileSync(dir+'/module.js', bravojsModule);
fs.writeFileSync(dir+'/package.dcp', packageText);
require('dcp-client').init().then(async () => {
  try
  {
    await require('dcp/publish').publish(dir+'/package.dcp');
    console.log('Model is published, may run a job using it');
  }
  catch (e) {/* publish wil log if error occurs, no need to double log */}
  fs.rmSync(dir, { recursive: true });
})
