\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{rjthesis}[2026/07/03 Journal of Software 2025 layout]

\LoadClass[twoside,UTF8]{ctexart}

\RequirePackage{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{xeCJK}
\RequirePackage{titlesec}
\RequirePackage{ifthen}
\RequirePackage{enumitem}
\RequirePackage{booktabs}
\RequirePackage{amsmath}
\RequirePackage{array}
\RequirePackage{caption}
\RequirePackage{etoolbox}
\RequirePackage{indentfirst}

\geometry{
  paperwidth=18.4cm,
  paperheight=26cm,
  asymmetric,
  inner=1.45cm,
  outer=1.45cm,
  top=0.6cm,
  bottom=3cm,
  includehead,
  headheight=18pt,
  headsep=18pt,
  footskip=1.2cm
}

% Font aliases used by older manuscripts.
\newcommand{\song}{\songti}
\newcommand{\fs}{\fangsong}
\newcommand{\kai}{\kaishu}
\newcommand{\hei}{\heiti}
\newcommand{\li}{\lishu}

% Journal sample sizes.
\newcommand{\chuhao}{\fontsize{42pt}{42pt}\selectfont}
\newcommand{\xiaochuhao}{\fontsize{36pt}{36pt}\selectfont}
\newcommand{\yihao}{\fontsize{28pt}{30pt}\selectfont}
\newcommand{\erhao}{\fontsize{21pt}{24pt}\selectfont}
\newcommand{\xiaoerhao}{\fontsize{18pt}{21pt}\selectfont}
\newcommand{\sanhao}{\fontsize{15.75pt}{19pt}\selectfont}
\newcommand{\sihao}{\fontsize{14pt}{17pt}\selectfont}
\newcommand{\xiaosihao}{\fontsize{12pt}{15pt}\selectfont}
\newcommand{\wuhao}{\fontsize{10.5pt}{13pt}\selectfont}
\newcommand{\xiaowuhao}{\fontsize{9pt}{12pt}\selectfont}
\newcommand{\liuhao}{\fontsize{7.875pt}{10pt}\selectfont}
\newcommand{\qihao}{\fontsize{5.25pt}{6pt}\selectfont}
\newcommand{\rjheaderfont}{\fontsize{7.5pt}{9pt}\selectfont}
\newcommand{\rjunitfont}{\fontsize{8pt}{9.5pt}\selectfont}
\newcommand{\rjcitefont}{\fontsize{8pt}{9.5pt}\selectfont}
\newcommand{\rjengabsfont}{\fontsize{7.5pt}{10pt}\selectfont}
\newcommand{\rjbibfont}{\fontsize{7.5pt}{10pt}\selectfont}
\newcommand{\rjbodyfont}{\songti\fontsize{9.5pt}{11pt}\selectfont}

\AtBeginDocument{%
  \rjbodyfont
  \setlength{\parindent}{2em}%
  \setlength{\parskip}{0pt}%
  \setlength{\emergencystretch}{2em}%
  \linespread{1}%
}

\setlist{nosep,leftmargin=2em,itemindent=0pt,labelsep=.5em}
\renewcommand{\theenumi}{(\arabic{enumi})}
\renewcommand{\labelenumi}{\theenumi}

% Metadata fields.
\newcommand{\@rjtitle}{}
\newcommand{\@rjauthor}{}
\newcommand{\@rjinfor}{}
\newcommand{\@rjetitle}{}
\newcommand{\@rjeauthor}{}
\newcommand{\@rjeinfor}{}
\newcommand{\@rjabstracttext}{}
\newcommand{\@rjkeywords}{}
\newcommand{\@rjeabstracttext}{}
\newcommand{\@rjekeywords}{}
\newcommand{\@rjclc}{}
\newcommand{\@rjcncite}{}
\newcommand{\@rjencite}{}
\newcommand{\@rjfund}{}
\newcommand{\@rjdates}{}
\newcommand{\@rjauthorsbio}{}
\newcommand{\@rjshorttitle}{}

\newcommand{\rjtitle}[1]{\renewcommand{\@rjtitle}{#1}}
\newcommand{\rjauthor}[1]{\renewcommand{\@rjauthor}{#1}}
\newcommand{\rjinfor}[1]{\renewcommand{\@rjinfor}{#1}}
\newcommand{\rjetitle}[1]{\renewcommand{\@rjetitle}{#1}}
\newcommand{\rjeauthor}[1]{\renewcommand{\@rjeauthor}{#1}}
\newcommand{\rjeinfor}[1]{\renewcommand{\@rjeinfor}{#1}}
\newcommand{\rjabstracttext}[1]{\renewcommand{\@rjabstracttext}{#1}}
\newcommand{\rjkeywords}[1]{\renewcommand{\@rjkeywords}{#1}}
\newcommand{\rjeabstracttext}[1]{\renewcommand{\@rjeabstracttext}{#1}}
\newcommand{\rjekeywords}[1]{\renewcommand{\@rjekeywords}{#1}}
\newcommand{\rjclc}[1]{\renewcommand{\@rjclc}{#1}}
\newcommand{\rjcncite}[1]{\renewcommand{\@rjcncite}{#1}}
\newcommand{\rjencite}[1]{\renewcommand{\@rjencite}{#1}}
\newcommand{\rjfund}[1]{\renewcommand{\@rjfund}{#1}}
\newcommand{\rjdates}[1]{\renewcommand{\@rjdates}{#1}}
\newcommand{\rjauthorsbio}[1]{\renewcommand{\@rjauthorsbio}{#1}}
\newcommand{\rjhead}[1]{\renewcommand{\@rjshorttitle}{#1}}
\newcommand{\rj}{软件学报}

\newcommand{\rj@ifnotempty}[2]{\ifx#1\@empty\else#2\fi}
\newcommand{\rj@frontpara}[2]{\noindent{#1 #2}\par}

\newcommand{\rjmaketitle}{%
  \thispagestyle{fancy}%
  \begingroup
  \setlength{\parindent}{0pt}%
  {\sihao\hei \@rjtitle\par}
  \vspace{8pt}
  {\xiaosihao\fs \@rjauthor\par}
  \vspace{8pt}
  {\rjunitfont\song \@rjinfor\par}
  \vspace{9pt}
  \rj@ifnotempty{\@rjabstracttext}{%
    {\xiaowuhao\noindent\hei 摘\hspace{2em}要:\hspace{.8em}\kai \@rjabstracttext\par}%
  }%
  \vspace{4pt}
  \rj@ifnotempty{\@rjkeywords}{%
    {\xiaowuhao\noindent\hei 关键词:\hspace{1em}\kai \@rjkeywords\par}%
  }%
  \vspace{4pt}
  \rj@ifnotempty{\@rjclc}{\rj@frontpara{\xiaowuhao 中图法分类号:}{\xiaowuhao \@rjclc}}%
  \rj@ifnotempty{\@rjcncite}{\rj@frontpara{\rjcitefont 中文引用格式:}{\rjcitefont \@rjcncite}}%
  \rj@ifnotempty{\@rjencite}{\rj@frontpara{\rjcitefont 英文引用格式:}{\rjcitefont \@rjencite}}%
  \vspace{8pt}
  \rj@ifnotempty{\@rjetitle}{{\wuhao\bfseries \@rjetitle\par}\vspace{5pt}}%
  \rj@ifnotempty{\@rjeauthor}{{\xiaowuhao \@rjeauthor\par}\vspace{3pt}}%
  \rj@ifnotempty{\@rjeinfor}{{\rjengabsfont \@rjeinfor\par}\vspace{5pt}}%
  \rj@ifnotempty{\@rjeabstracttext}{{\rjengabsfont\noindent\textbf{Abstract}: \@rjeabstracttext\par}\vspace{3pt}}%
  \rj@ifnotempty{\@rjekeywords}{{\rjengabsfont\noindent\textbf{Key words}: \@rjekeywords\par}}%
  \vspace{10pt}
  \endgroup
}

% Back matter blocks.
\newcommand{\rjappendixinfo}{%
  \par\vspace{8pt}
  \rj@ifnotempty{\@rjauthorsbio}{%
    {\xiaowuhao\noindent\hei 作者简介\par}
    {\rjbibfont \@rjauthorsbio\par}
  }%
  % \rj@ifnotempty{\@rjfund}{%
  %   \vspace{4pt}{\rjbibfont\noindent\textsuperscript{*}\quad 基金项目: \@rjfund\par}%
  % }%
  % \rj@ifnotempty{\@rjdates}{%
  %   {\rjbibfont\noindent 收稿时间: \@rjdates\par}%
  % }%
}

% Legacy environment kept for older sources; new papers should use \rjabstracttext.
\newenvironment{rjabstract}{%
  \par\noindent\xiaowuhao\hei 摘\hspace{2em}要:\hspace{.8em}\kai
}{\par}

\titleformat{\section}{\heiti\wuhao}{\thesection}{1em}{}
\titlespacing*{\section}{0pt}{8pt}{6pt}
\titleformat{\subsection}{\heiti\xiaowuhao}{\thesubsection}{1em}{}
\titlespacing*{\subsection}{0pt}{5pt}{3pt}
\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}

\captionsetup{font={small},labelfont=normal,labelsep=quad,skip=4pt}
\AtBeginEnvironment{tabular}{\xiaowuhao}
\AtBeginEnvironment{thebibliography}{%
  \rjbibfont
  \setlength{\itemsep}{0pt}%
  \setlength{\parsep}{0pt}%
}
\renewcommand{\refname}{References}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyhead[LO]{\ifthenelse{\value{page}=1}{\rjheaderfont 软件学报ISSN 1000-9825, CODEN RUXUEW\\\textit{Journal of Software}, [doi: 10.13328/j.cnki.jos.000000]\\\copyright 中国科学院软件研究所版权所有.}{\xiaowuhao\kai \@rjshorttitle}}
\fancyhead[RO]{\ifthenelse{\value{page}=1}{\rjheaderfont E-mail: jos@iscas.ac.cn\\http://www.jos.org.cn\\Tel: +86-10-62562563}{\thepage}}
\fancyhead[LE]{\thepage}
\fancyhead[RE]{\xiaowuhao\textit{Journal of Software} \kai 软件学报}
\setlength{\headheight}{36pt}
