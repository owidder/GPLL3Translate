const pdf = require('html-pdf');
const fs = require('fs');
const path = require('path');

(async () => {
  const folderPath = '/Users/oliverwidder/dev/github/GPLL3Translate/tables'; // Replace with your folder path
  const files = fs.readdirSync(folderPath).filter(file => file.endsWith('.html'));

  files.forEach(file => {
    const filePath = path.join(folderPath, file);
    const html = fs.readFileSync(filePath, 'utf8');
    const outputFilePath = path.join(folderPath, `${path.basename(file, '.html')}.pdf`);

    const options = {
      format: 'A4',
      orientation: 'landscape',
      border: {
        top: '0.5in',
        right: '0.5in',
        bottom: '0.5in',
        left: '0.5in'
      },
      type: 'pdf',
      quality: '100',
      renderDelay: 1000,
      timeout: 30000
    };

    pdf.create(html, options).toFile(outputFilePath, (err, res) => {
      if (err) return console.log(err);
      console.log(res);
    });
  });
  const folderPath = '/Users/oliverwidder/dev/github/GPLL3Translate/tables'; // Replace with your folder path
  const files = fs.readdirSync(folderPath).filter(file => file.endsWith('.html'));

  for (const file of files) {
    const filePath = path.join(folderPath, file);
    const page = await browser.newPage();
    await page.goto(`file://${filePath}`, { waitUntil: 'networkidle2' });

    // Adjust the table and page to fit the content
    await page.addStyleTag({
      content: `
        table {
          width: 100%;
        }
      `
    });

    // Generate the PDF with landscape orientation
    const outputFilePath = path.join(folderPath, `${path.basename(file, '.html')}.pdf`);
    await page.pdf({
      path: outputFilePath,    // Output file path
      format: 'A4',            // Paper format
      landscape: true,         // Landscape orientation
      printBackground: true,   // Print background graphics
    });

    await page.close();
  }

  await browser.close();
})();
