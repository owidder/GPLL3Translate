const puppeteer = require('puppeteer');
const path = require('path');

async function convertHtmlToPdf(htmlFilePath, outputPdfPath) {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // Get the absolute path of the HTML file
  const absoluteHtmlFilePath = path.resolve(htmlFilePath);

  // Load the HTML file
  await page.goto(`file://${absoluteHtmlFilePath}`, {
    waitUntil: 'networkidle0'
  });

  // Generate PDF
  await page.pdf({
    path: outputPdfPath,
    format: 'A4',
    printBackground: true,
    margin: {
      top: '20px',
      right: '20px',
      bottom: '20px',
      left: '20px'
    }
  });

  await browser.close();
  console.log(`PDF created successfully: ${outputPdfPath}`);
}

// Usage
const htmlFilePath = 'aider/01_co2_pricing.json_table.de_sv.html';
const outputPdfPath = 'aider/01_co2_pricing.json_table.de_sv.pdf';

convertHtmlToPdf(htmlFilePath, outputPdfPath).catch(console.error);
