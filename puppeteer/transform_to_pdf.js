const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // Replace with the path to your HTML file
  await page.goto('file:///Users/oliverwidder/dev/github/GPLL3Translate/tables/01_co2_pricing.json_table.de_sv.html', { waitUntil: 'networkidle2' });

  // Adjust the table and page to fit the content
  await page.addStyleTag({
    content: `
      table {
        width: 100%;
      }
    `
  });

  // Generate the PDF with landscape orientation
  await page.pdf({
    path: 'output.pdf',        // Output file path
    format: 'A4',              // Paper format
    landscape: true,           // Landscape orientation
    printBackground: true,     // Print background graphics
  });

  await browser.close();
})();
