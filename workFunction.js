/*
 *  @file   workFunction.js
 *
 *  @author Erin Peterson, erin@distributive.network
 *  @author Mehedi Arefin, mehedi@distributive.network
 *  @date   Oct 25th, 2022
 */
'use strict';

/**
 * Work function that is passed to DCP Worker
 * @async
 * @function
 * @name workFunction
 * @param { object } sliceData
 * @param { object } labels
 */
async function workFunction(sliceData, labels, preprocess, postprocess, pythonPackages, modelArg) {
	progress(0);
	require('dcp-wasm.js');
  debugger;
  var model, packages, preStr, postStr;
  if (!modelArg)
  {
    const moduleInput = require('module.js');
    model    = b64ToArrayBuffer(moduleInput.model);
    packages = moduleInput.packages;
    preStr   = atob(moduleInput.preprocess);
    postStr  = atob(moduleInput.postprocess);
  }
  else
  {
    model    = b64ToArrayBuffer(modelArg);
    preStr   = preprocess;
    postStr  = postprocess;
    packages = pythonPackages;

  }

	// DECLARE VARIABLES
	let feeds       = {};
	let finalResult = {};
	let infResult   = {};
	const numInputs = Object.keys(sliceData.b64Data).length;



	// CREATE ORT SESSION
	progress(0.1);

	if (!globalThis.ort) {
		globalThis.ort = require('dcp-ort.js');
	};

	ort.env.wasm.simd = true;
  console.log(ort.env.versions);

	// add else -> comments get run
	if (!globalThis.session) {
    globalThis.session = await ort.InferenceSession.create(model, {
      executionProviders    : [labels['webgpu'] ? 'webgpu' : 'wasm'],
      graphOptimizationLevel: 'all'
    });
	}

	const inputNames  = session.inputNames;
	const outputNames = session.outputNames;

	progress(0.2);
	let _progress = 0.2;

	// DECLARE UTIL FUNCTIONS
	/**
	 * Convert B64 to Array Buffer in DCP work function
	 * @function
	 * @name b64ToArrayBuffer
	 * @param { string } base64
	 * @returns { Uint8Array.buffer }
	 */
	function b64ToArrayBuffer(base64) {
		const binary = atob(base64);
		const len    = binary.length;
		const bytes  = new Uint8Array(len);

		for (let i = 0; i < len; i++) {
			bytes[i] = binary.charCodeAt(i);
		}

		return bytes.buffer;
	};

	/**
	 * Map inference results from a list of key-value pairs into an object in DCP work function
	 * @function
	 * @name mapToObj
	 * @param  { Map<string, string>} m - Map which is to be converted to an object
	 * @returns { object } obj
	 */
	function mapToObj(m) {
		const obj = Object.fromEntries(m);
		for (const key of Object.keys(obj)) {
			if (obj[ key ].constructor.name == 'Map') {
				obj[ key ] = mapToObj(obj[ key ]);
			}
		}
		return obj;
	};

	/**
	 * Performs the inference on the feeds
	 * @async
	 * @function
	 * @name runInference
	 * @param { object } feeds
	 * @returns { Promise<object> } infOut
	 */
	async function runInference(feeds) {
		const infStart = performance.now();
		const infOut   = await session.run(feeds);
		return infOut;
	};

	/**
	 * Python Work Function
	 * @async
	 * @function
	 * @name pythonWorkFunction
	 * @param { Array<string> }packages
	 * @param { object } sliceData
	 * @param { object } labels
	 * @returns { object } finalResult
	 */
	async function pythonWorkFunction(packages, sliceData, labels) {
		// GET PYODIDE CORE
		if (!globalThis.pyodideCore) {
			globalThis.pyodideCore = require('pyodide-core.js');
		}

		if (!globalThis.pyodide) {
			globalThis.pyodide = await pyodideCore.pyodideInit();
		}

		// PREP PYTHON PACKAGES
		await pyodideCore.loadPackage(packages);

		// PUT STRINGS IN PYTHON SPACE
		globalThis.preStr  = preStr;
		globalThis.postStr = postStr;

		// WRITE PRE AND POST PROCESS FUNCTIONS TO DISK
		pyodide.runPython(`
import js

with open('./preprocess.py', 'w') as f:
  f.write( js.globalThis.preStr )

with open('./postprocess.py', 'w') as f:
  f.write( js.globalThis.postStr )
    `);

		// IMPORT PRE AND POST PROCESS FUNCTIONS
		try {
			preprocess = pyodide.runPython(`
import preprocess 

preprocessFunction = [ i for i in dir(preprocess) if 'preprocess' == i.lower() ]
pythonPre = getattr( preprocess, preprocessFunction[0] )
pythonPre
      `);

			postprocess = pyodide.runPython(`
import postprocess 

postprocessFunction = [ i for i in dir(postprocess) if 'postprocess' in i.lower() ];
pythonPost = getattr( postprocess, postprocessFunction[0] )
pythonPost
        `);
		} catch(error) {
			const stack    = error.message.split('\n');
			const errorMsg = stack[stack.length - 2];
      console.log('ahhhhhhhhhhhh error', error)
			return {
				'code': 'pyodide',
				'msg' : errorMsg,
				'file': labels['fileID']
			};
		}

		// CORE LOOP
		for (const [ key, value ] of Object.entries(sliceData.b64Data)) {
			progress(_progress);
			start = performance.now();

			// DECODE INPUT
			labels['fileID'] = key;
			const b64Input   = value;
			const abInput    = b64ToArrayBuffer(b64Input);

			// CONVERT AB TO BYTES AND RUN PYTHON PREPROCESS
			pyodide.globals.set('preprocessArgs', [abInput, inputNames]);
			const preStart = performance.now();

			try {
				pyPreOut = pyodide.runPython(`
import numpy as np
preprocessArgs = preprocessArgs.to_py()
bytesInput = preprocessArgs[0].tobytes()
inputNames = preprocessArgs[1]
inputNames = np.array(inputNames)
feed = pythonPre(bytesInput, inputNames)
for(key, array) in feed.items():
	feed[key] = np.ascontiguousarray(array, dtype=array.dtype)
feed
        `);
			} catch(error) {

				const stack    = error.message.split('\n');
				const errorMsg = stack[stack.length - 2];

				return {
					'code': 'preprocess',
					'msg' : errorMsg,
					'file': labels['fileID']
				};
			}

			pyodide.globals.pop('preprocessArgs');

			// CONVERT NP ARRAY TO ONNX TENSOR
			for (const key of pyPreOut.keys()) {
				const value  = pyPreOut.get(key);
				feeds[ key ] = new ort.Tensor(
					value.dtype.name,
					value.getBuffer().data,
					value.shape.toJs());
			};

			// RUN INFERENCE
			try {
				infResult = await runInference(feeds);
				feeds     = {};
			} catch(error) {
				return {
					'code': 'inference',
					'msg' : error.message,
					'file': labels['fileID']
				};
			}

			for (const key of Object.keys(infResult)) {
				const value       = infResult[key];
				value.data_buffer = value.data.buffer;
				infResult[key]    = value;
			};

			const postStart = performance.now();
			pyodide.globals.set('postprocessArgs', [infResult, labels, outputNames]);

			try {
				infResult = pyodide.runPython(`
import numpy as np

postprocessArgs = postprocessArgs.to_py()

outData = postprocessArgs[0]
labels  = postprocessArgs[1]
outputNames = postprocessArgs[2]
for key, value in outData.items():
  outDims   = value.dims.to_py()
  outType = str(value.type)
  pyOutData = value.data_buffer
  pyOutData = pyOutData.to_py()
  outData[key] = np.frombuffer(pyOutData.tobytes(), dtype = outType).reshape(outDims)
pyOut = pythonPost(outData, labels, outputNames)
pyOut
      `);
			} catch(error) {

				const stack    = error.message.split('\n');
				const errorMsg = stack[stack.length - 2];

				return {
					'code': 'postprocess',
					'msg' : errorMsg,
					'file': labels['fileID']
				};
			}

			infResult = infResult.toJs();
			infResult = mapToObj(infResult);

			finalResult[ key ] = infResult;
			_progress = _progress + (.8 / numInputs);
		}

		return finalResult;
	};

	finalResult = await pythonWorkFunction(packages, sliceData, labels);

	// RETURN RESULTS
	progress(1);
	return finalResult;
}

exports.workFunction = workFunction;

