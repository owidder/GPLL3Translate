const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

async function convertHtmlToPdf(htmlFilePath, outputPdfPath) {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // Get the absolute path of the HTML file
  const absoluteHtmlFilePath = path.resolve(htmlFilePath);

  // Read the HTML file
  const htmlContent = fs.readFileSync(absoluteHtmlFilePath, 'utf8');

  // Set the HTML content
  await page.setContent(htmlContent, {
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

const htmlFilePath = path.join(__dirname, '01_co2_pricing.json_table.de_sv.html');
const outputPdfPath = path.join(__dirname, '01_co2_pricing.json_table.de_sv.pdf');

convertHtmlToPdf(htmlFilePath, outputPdfPath).catch(console.error);
