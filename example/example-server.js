/**
 * @file   example-server.js
 *         Example web server for hosting model to be fetched by workers
 *
 * @author Ryan Saweczko, ryansaweczko@distributive.network
 */

const express = require('express');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

var modelInfo;
try
{
  modelInfo = JSON.parse(fs.readFileSync('model.json', { encoding: 'utf-8' }));
}
catch (error)
{
  console.error('Unable to find model.json file in directory, exiting');
  process.exit(1);
}

const app = express();
const port = 3500;
app.use(cors());

const fileDirectory = path.join(__dirname);

app.get('/:filename', (req, res) => {
  const { filename } = req.params;
  const file = path.join(fileDirectory, filename);
  if (filename === modelInfo.model)
  {
    const content = fs.readFileSync(file).toString('base64');
    res.send(content);
  }
  else if ([modelInfo.preprocess, modelInfo.postprocess].includes(filename))
  {
    res.sendFile(file, (err) => {
      if (err)
        res.status(500).send('Error sending the file.');
    });
  }
  else
  {
    console.error(`request for invalid file: ${filename}`);
    res.status(404).send('Invalid file');
  }
});

// Start server
app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
