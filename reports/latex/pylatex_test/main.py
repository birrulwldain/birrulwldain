from pylatex import Document, Section, Subsection, Command, Figure, NoEscape, Package, Tabular, Math, Alignat, Itemize
from pylatex.utils import italic, NoEscape

def generate_complex_latex_document():
    doc = Document(documentclass='article',
                   document_options=['10pt', 'a4paper', 'twocolumn'],
                   fontenc='T1',
                   inputenc='utf8')

    doc.append(Command('author', 'Birrul Wildan'))
    doc.append(Command('title', 'A More Complex PyLaTeX Document Example'))
    doc.append(Command('date', NoEscape(r'\today')))

    # Add packages
    doc.packages.append(Package('graphicx'))
    doc.packages.append(Package('amsmath'))
    doc.packages.append(Package('amssymb'))
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('hyperref'))

    # Add title page
    doc.append(NoEscape(r'\maketitle'))

    # Abstract
    with doc.create(Section('Abstract')):
        doc.append('This document demonstrates a more complex usage of PyLaTeX, '
                   'including sections, subsections, figures, tables, lists, '
                   'and mathematical equations. It serves as an example for '
                   'generating structured LaTeX documents programmatically.')

    # Introduction Section
    with doc.create(Section('Introduction')):
        doc.append('Welcome to this advanced PyLaTeX example. '
                   'We will explore various features to create a rich document.')

        with doc.create(Subsection('Document Structure')):
            doc.append('LaTeX documents are structured using sections and subsections.')

    # Figures Section
    with doc.create(Section('Figures')):
        doc.append('Here is an example of including a figure. '
                   'Please ensure you have an image file named `example-image-a.png` '
                   'in the same directory for this to compile correctly, '
                   'or replace it with an existing image path.')
        with doc.create(Figure(position='h!')) as figure:
            figure.add_image('example-image-a.png', width=NoEscape(r'0.8\textwidth'))
            figure.add_caption('A placeholder image demonstrating figure inclusion.')
            figure.append(Command('label', 'fig:example_image'))

    # Tables Section
    with doc.create(Section('Tables')):
        doc.append('Below is an example of a simple table.')
        with doc.create(Tabular('c|c|c')) as table:
            table.add_hline()
            table.add_row(('Header 1', 'Header 2', 'Header 3'))
            table.add_hline()
            table.add_row(('Row 1 Col 1', 'Row 1 Col 2', 'Row 1 Col 3'))
            table.add_row(('Row 2 Col 1', 'Row 2 Col 2', 'Row 2 Col 3'))
            table.add_hline()
        doc.append(Command('label', 'tab:example_table'))

    # Lists Section
    with doc.create(Section('Lists')):
        doc.append('Here are some examples of lists:')
        with doc.create(Itemize()) as itemize:
            itemize.add_item('First item in the list.')
            itemize.add_item('Second item, which is a bit longer to show wrapping.')
            itemize.add_item('Third item.')

    # Mathematics Section
    with doc.create(Section('Mathematics')):
        doc.append('We can include inline math like $E=mc^2$ or displayed equations:')
        doc.append(NoEscape(r'\[ \alpha + \beta = \gamma \]'))
        doc.append('And aligned equations:')
        with doc.create(Alignat(numbering=True, escape=False)) as alignat:
            alignat.append(r'f(x) &= x^2 + 2x + 1 \\')
            alignat.append(r'     &= (x+1)^2')
        doc.append(Command('label', 'eq:quadratic'))

    # Conclusion Section
    with doc.create(Section('Conclusion')):
        doc.append('This document has showcased various features of PyLaTeX. '
                   'You can refer to Figure \ref{fig:example_image} and Table \ref{tab:example_table}. '
                   'Equation \ref{eq:quadratic} is an example of aligned math.')
        doc.append(NoEscape(r'\newline'))
        doc.append(italic('Happy LaTeXing with Python!'))

    doc.generate_pdf('my_complex_document', clean_tex=True)
    print('PDF generated: my_complex_document.pdf')

if __name__ == '__main__':
    generate_complex_latex_document()
