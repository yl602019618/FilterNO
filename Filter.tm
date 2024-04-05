<TeXmacs|2.1.2>

<style|<tuple|generic|chinese>>

<\body>
  <doc-data|<doc-title|Neural Operator for Nonlinear
  Filter>|<doc-author|<author-data|<author-name|zhijun
  zeng>|<\author-affiliation>
    <date|>
  </author-affiliation>>>>

  <section|Problem setting>

  Consider the signal based model

  <\equation*>
    <around*|{|<tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|l>|<cwith|1|-1|1|1|cell-lborder|0ln>|<cwith|1|-1|1|1|cell-rborder|0ln>|<table|<row|<cell|d*x<rsub|t>=f<around*|(|x<rsub|t>,t|)>*d*t+g<around*|(|x<rsub|t>,t|)>*d*v<rsub|t>>>|<row|<cell|d*y<rsub|t>=h<around*|(|x<rsub|t>,t|)>*d*t+d*w<rsub|t>>>>>>|\<nobracket\>>
  </equation*>

  where <math|x<rsub|t>\<in\>\<bbb-R\><rsup|n>> is the state of system at
  time t, the initial state <math|x<rsub|0>\<sim\>p<rsub|0>> satisfying some
  initial distribution. The observation <math|y<rsub|t>\<in\>\<bbb-R\><rsup|m>>,
  initialize at <math|y<rsub|0>=0>. The noise is vector based Brownian motion
  <math|\<bbb-E\><around*|[|dv<rsub|t>dv<rsub|t><rsup|\<Tau\>>|]>=Q<around*|(|t|)>dt\<in\>\<bbb-R\><rsup|n\<times\>n>>,
  and <math|> <math|\<bbb-E\><around*|[|dw<rsub|t>dw<rsub|t><rsup|\<Tau\>>|]>=S<around*|(|t|)>dt\<in\>\<bbb-R\><rsup|n\<times\>n>>,
  here <math|S<around*|(|t|)>\<gtr\>0>.

  The unnormalized conditional density function of the states
  <math|x<rsub|t>>, denoted by <math|\<sigma\><around*|(|x,t|)>> satisfies
  the following SPDE (DMZ)

  <\equation*>
    <around*|{|<tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|l>|<cwith|1|-1|1|1|cell-lborder|0ln>|<cwith|1|-1|1|1|cell-rborder|0ln>|<table|<row|<cell|d*\<sigma\><around|(|x,t|)>=\<cal-L\>*\<sigma\><around|(|x,t|)>*d*t+\<sigma\><around|(|x,t|)>*h<rsup|T><around|(|x,t|)>*S<rsup|-1>*d*y<rsub|t>>>|<row|<cell|\<sigma\><around|(|x,0|)>=\<sigma\><rsub|0><around|(|x|)>>>>>>|\<nobracket\>>
  </equation*>

  where <math|\<sigma\><around*|(|x,0|)>=p<rsub|0><around*|(|x|)>>, and\ 

  <\equation*>
    \<cal-L\><around|(|\<cdummy\>|)>\<assign\><frac|1|2>*<big|sum><rsub|i,j=1><rsup|n><frac|\<partial\><rsup|2>|\<partial\>*x<rsub|i>*\<partial\>*x<rsub|j>>*<around*|(|<around*|(|g*Q*g<rsup|T>|)><rsub|i*j>\<cdot\>|)>-<big|sum><rsub|i=1><rsup|n><frac|\<partial\>*<around*|(|f<rsub|i>\<cdot\>|)>|\<partial\>*x<rsub|i>>
  </equation*>

  Here we call <math|f> the drift term, <math|g> the diffusion term, <math|h>
  the observation term.

  By an invertible exponential transformation\ 

  <\equation*>
    \<sigma\><around|(|x,t|)>=exp <around*|(|h<rsup|T><around|(|x,t|)>*S<rsup|-1><around|(|t|)>*y<rsub|t>|)>*u<around|(|x,t|)>
  </equation*>

  We obtain an PDE with stochastic coefficients

  <\equation*>
    <around*|{|<\aligned>
      <tformat|<table|<row|<cell|<frac|\<partial\>|\<partial\>*t>*u<around|(|x,t|)>+>|<cell|<frac|\<partial\>|\<partial\>*t>*<around*|(|h<rsup|T>*S<rsup|-1>|)>*y<rsub|t>*u<around|(|x,t|)>=>>|<row|<cell|>|<cell|exp
      <around*|(|-h<rsup|T>*<around|(|x,t|)>*S<rsup|-1><around|(|t|)>*y<rsub|t>|)>*<around*|(|\<cal-L\>-<frac|1|2>*h<rsup|T>*S<rsup|-1>*h|)>>>|<row|<cell|>|<cell|\<cdot\><around*|(|exp
      <around*|(|-h<rsup|T>*<around|(|x,t|)>*S<rsup|-1><around|(|t|)>*y<rsub|t>|)>*u<around|(|x,t|)>|)>>>|<row|<cell|u<around|(|x,0|)>=>|<cell|\<sigma\><rsub|0><around|(|x|)>>>>>
    </aligned>|\<nobracket\>>
  </equation*>

  This equation is called pathwise robust DMZ equation.

  \;

  Assume that the observation time sequences
  <math|0=t<rsub|0>\<less\>\<cdots\>\<less\>t<rsub|N<rsub|t>>=T> are given.
  Then the observation <math|<around*|{|y<rsub|t<rsub|j>>|}>> at each
  observation time <math|t<rsub|j>> are unknown until the online experiment.
  Therefore, one can freeze the <math|y<rsub|t>> to
  <math|y<rsub|t<rsub|j-1>>> in time <math|t<rsub|j-1>\<leqslant\>t\<less\>t<rsub|j>>,
  making the exponential transformation\ 

  <\equation*>
    <math|u<rsub|j><around|(|x,t|)>=exp <around*|(|h<rsup|T><around|(|x,t|)>*S<rsup|-1><around|(|t|)>*y<rsub|t<rsub|j-1>>|)>*u<around|(|x,t|)>>.
  </equation*>

  One obtain that <math|u<rsub|j>> satisfies the FKE

  <\equation*>
    <frac|\<partial\>|\<partial\>*t>*u<rsub|j><around|(|x,t|)>=<around*|(|\<cal-L\>-<frac|1|2>*h<rsup|T>*S<rsup|-1>*h|)>*u<rsub|j><around|(|x,t|)>
  </equation*>

  \;

  \;

  The finite difference solver:\ 

  M. Yueh, W. Lin, and S. T. Yau, \PAn efficient num<nbhyph>erical method\ 

  for solving high-dimensional nonlinear filtering problems,\Q Com\B

  mun. Inf. Syst., vol. 14, no. 4, pp. 243-262, 2014.

  <section|Numerical Method for NLS>

  For\ 

  <\equation*>
    <around*|{|<tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|l>|<cwith|1|-1|1|1|cell-lborder|0ln>|<cwith|1|-1|1|1|cell-rborder|0ln>|<table|<row|<cell|d*x<rsub|t>=f<around*|(|x<rsub|t>,t|)>*d*t+*d*v<rsub|t>>>|<row|<cell|d*y<rsub|t>=h<around*|(|x<rsub|t>,t|)>*d*t+d*w<rsub|t>>>>>>|\<nobracket\>>
  </equation*>

  The FKE becomes

  <\equation*>
    <frac|\<partial\>|\<partial\>*t>*u<rsub|j><around|(|x,t|)>=<frac|1|2>\<Delta\>*u<rsub|j><around|(|x,t|)>-f<around*|(|x|)>\<cdot\>\<nabla\>u<rsub|j><around*|(|x,t|)>-<around*|(|\<nabla\>\<cdot\>f<around*|(|x|)>+<frac|1|2><around*|\<\|\|\>|h<around*|(|s|)>|\<\|\|\>><rsub|2><rsup|2>|)>u<rsub|j><around*|(|x,t|)>
  </equation*>

  The initial condition is <math|u<around*|(|0,s|)>=\<sigma\><rsub|0><around*|(|s|)>>,
  once the new measurement <math|y<around*|(|t<rsub|k>|)>> arrive we correct
  the solution <math|<around*|(|u<rsub|j><around*|(|x,t<rsub|k>|)>|)>> at
  <math|t<rsub|k>> by

  <\equation*>
    exp<around*|{|<around*|[|y<around*|(|t<rsub|k>|)>-y<around*|(|t<rsub|k-1>|)>|]>\<cdot\>h<around*|(|x|)>|}>u<rsub|j><around*|(|x,t<rsub|k>|)>
  </equation*>

  Initially, we discretize the time interval
  <math|<around*|[|t<rsub|k-1>,t<rsub|k>|]>> by taking a uniform partition
  <math|<around*|[|t<rsub|k-1>,t<rsub|k-1>+\<Delta\>t,\<ldots\>t<rsub|k-1>+N\<Delta\>t=t<rsub|k>|]>>.

  For the sake of computations, we restrict the domain
  <math|\<bbb-R\><rsup|D>> to <math|<around*|[|-R,R|]><rsup|D> cells> and
  partition it uniformly.

  <subsection|2D example>

  <\equation*>
    <around*|{|<tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|l>|<cwith|1|-1|1|1|cell-lborder|0ln>|<cwith|1|-1|1|1|cell-rborder|0ln>|<table|<row|<cell|d*x<rsub|1>=cos<around*|(|x<rsub|1>|)>d*v<rsub|1>>>|<row|<cell|d*x<rsub|2>=cos<around*|(|x<rsub|2>|)>d*v<rsub|2>>>|<row|<cell|d*y<rsub|1>=x<rsub|1><rsup|3>*d*t+d*w<rsub|1>>>|<row|<cell|d*y<rsub|2>=x<rsub|2><rsup|3>*d*t+d*w<rsub|2>>>>>>|\<nobracket\>>
  </equation*>

  Then\ 

  <\equation*>
    f<around*|(|x|)>=<around*|[|<tabular*|<tformat|<table|<row|<cell|cos<around*|(|x<rsub|1>|)>>>|<row|<cell|cos<around*|(|x<rsub|2>|)>>>>>>|]>,h<around*|(|x|)>=<around*|[|<tabular*|<tformat|<table|<row|<cell|x<rsub|1><rsup|3>>>|<row|<cell|x<rsub|2><rsup|3>>>>>>|]>
  </equation*>

  The strong form of equation will be\ 

  <\equation*>
    <frac|\<partial\>|\<partial\>*t>*u<rsub|j>=<frac|1|2>\<Delta\>*u<rsub|j>-cos<around*|(|x<rsub|1>|)>\<partial\><rsub|x<rsub|1>>u<rsub|j>-cos<around*|(|x<rsub|2>|)>\<partial\><rsub|x<rsub|2>>u<rsub|j>+<around*|(|sin<around*|(|x<rsub|1>|)>+sin<around*|(|x<rsub|2>|)>-<frac|1|2><around*|(|x<rsub|1><rsup|6>+x<rsub|2><rsup|6>|)>|)>u<rsub|j>
  </equation*>

  The weak form of equation will be\ 

  <\equation*>
    <frac|\<less\>u<rsub|j><around*|(|t+\<Delta\>t|)>,v\<gtr\>-\<less\>u<rsub|j><around*|(|t|)>,v\<gtr\>|\<Delta\>t>=-<frac|1|2>\<less\>\<nabla\>u<rsub|j>,\<nabla\>v\<gtr\>-\<less\>f\<cdot\>\<nabla\>u<rsub|j>,v\<gtr\>+\<less\>a<around*|(|x|)>u<rsub|j>,v<rsub|>\<gtr\>
  </equation*>

  where <math|a<around*|(|x|)>=sin<around*|(|x<rsub|1>|)>+sin<around*|(|x<rsub|2>|)>-<frac|1|2><around*|(|x<rsub|1><rsup|6>+x<rsub|2><rsup|6>|)>>.
  The boundary condition is dirichlet boundary condition.

  \;

  Here is some discretization parameter: <math|\<Delta\>t=0.001,\<sigma\><rsub|0>(s)
  = exp {\<minus\>10<around*|\||x|\|><rsup|2>},h=0.1,\<Omega\>=<around*|[|-5,5|]><rsup|2>>.

  <section|Numerical Example>

  <subsection|1D>

  Consider the problem of\ 

  <\equation*>
    <around*|{|<tabular*|<tformat|<table|<row|<cell|dx<rsub|1>=dv<rsub|1>>>|<row|<cell|dy<rsub|1>=x<rsub|1><around*|(|1+0.2cos<around*|(|x<rsub|1>|)>|)>dt+dw<rsub|1>>>>>>|\<nobracket\>>
  </equation*>

  where <math|\<bbb-E\><around*|[|dw<rsub|t>dw<rsub|t>|]>=I<rsub|1>dt,><math|\<bbb-E\><around*|[|dv<rsub|t>dv<rsub|t>|]>=0.1I<rsub|1><rsub|>dt>.
  For filter problem, <math|x<around*|(|0|)>=<around*|[|1.0|]>>.\ 

  For operator learning problem, <math|g<around*|(|t|)>=1,h<around*|(|x,t|)>=x<around*|(|1+0.2cos<around*|(|x|)>|)>,Q<around*|(|t|)>=0.1,S<around*|(|t|)>=1>,
  the FKE becomes

  <\equation*>
    <frac|\<partial\>|\<partial\>t>u<rsub|j><around*|(|x,t|)>=<around*|(|<frac|1|2>*0.1<frac|\<partial\><rsup|2>u<rsub|j>|\<partial\>*x<rsub|><rsup|2>>-<frac|1|2>*x<rsup|2><around*|(|1+0.2cos<around*|(|x|)>|)><rsup|2>u<rsub|j><around*|(|x|)>|)>
  </equation*>

  The neural operator problem can be defined as predicting
  <math|u<rsub|j><around*|(|x,t+\<Delta\>t|)> >from
  <math|u<rsub|j><around*|(|x,t|)>>.

  <subsection|2D>

  Consider the problem of\ 

  <\equation*>
    <around*|{|<tabular*|<tformat|<cwith|1|-1|1|1|cell-halign|l>|<cwith|1|-1|1|1|cell-lborder|0ln>|<cwith|1|-1|1|1|cell-rborder|0ln>|<table|<row|<cell|d*x<rsub|1>=d*v<rsub|1>>>|<row|<cell|d*x<rsub|2>=d*v<rsub|2>>>|<row|<cell|d*y<rsub|1>=x<rsub|1>*<around*|(|1+0.2*cos
    <around*|(|x<rsub|2>|)>|)>*d*t+d*w<rsub|1>>>|<row|<cell|d*y<rsub|2>=x<rsub|2>*<around*|(|1+0.2*cos
    <around*|(|x<rsub|1>|)>|)>*d*t+d*w<rsub|2>>>>>>|\<nobracket\>>
  </equation*>

  where <math|\<bbb-E\><around*|[|dw<rsub|t>dw<rsub|t>|]>=I<rsub|2>dt,><math|\<bbb-E\><around*|[|dv<rsub|t>dv<rsub|t>|]>=0.1I<rsub|2><rsub|>dt>.
  For filter problem, <math|x<around*|(|0|)>=<around*|[|1.0,1.2|]><rsup|\<top\>>>.\ 

  For operator learning problem, <math|g<around*|(|t|)>=I<rsub|2>,h<around*|(|x,t|)>=<around*|[|<tabular*|<tformat|<table|<row|<cell|x<rsub|1>*<around*|(|1+0.2*cos
  <around*|(|x<rsub|2>|)>|)>>>|<row|<cell|x<rsub|2>*<around*|(|1+0.2*cos
  <around*|(|x<rsub|1>|)>|)>>>>>>|]>,Q<around*|(|t|)>=0.1I<rsub|2>,S<around*|(|t|)>=I<rsub|2>>,
  the FKE becomes

  <\equation*>
    g*Q*g<rsup|T>=0.1I<rsub|2>,h<rsup|T>*S<rsup|-1>*h=h<rsup|T>h=
  </equation*>

  <\equation*>
    <frac|\<partial\>|\<partial\>t>u<rsub|j><around*|(|x<rsub|1>,x<rsub|2>,t|)>=<frac|1|2>*0.1<around*|(|<frac|\<partial\><rsup|2>u<rsub|j>|\<partial\>*x<rsub|1><rsub|><rsup|2>>+<frac|\<partial\><rsup|2>u<rsub|j>|\<partial\>*x<rsub|2><rsub|><rsup|2>>|)>-<around*|[|x<rsub|1><rsup|2>*<around*|(|1+0.2*cos
    <around*|(|x<rsub|2>|)>|)><rsup|2>+x<rsub|2><rsup|2>*<around*|(|1+0.2*cos
    <around*|(|x<rsub|1>|)>|)><rsup|2>|]>u<rsub|j><around*|(|x<rsub|1>,x<rsub|2>,t|)>
  </equation*>

  The neural operator problem can be defined as predicting
  <math|u<rsub|j><around*|(|x,t+\<Delta\>t|)> >from
  <math|u<rsub|j><around*|(|x,t|)>>.

  <subsection|Post-process>

  Obtained <math|u<rsub|j><around*|(|x,t+\<Delta\>t|)>>, we can compute\ 

  <\equation*>
    *u<around|(|x,t+\<Delta\>t|)>=exp <around*|(|-h<rsup|T><around|(|x,t|)>*S<rsup|-1><around|(|t|)>*y<rsub|t<rsub|j-1>>|)>u<rsub|j><around*|(|x,t+\<Delta\>t|)>
  </equation*>

  And the prediction of <math|\<sigma\><around*|(|x,t|)>> is\ 

  <\equation*>
    \<sigma\><around|(|x,t+\<Delta\>t|)>=exp
    <around*|(|h<rsup|T><around|(|x,t|)>*S<rsup|-1><around|(|t|)>*y<rsub|t<rsub|j>>|)>*u<around|(|x,t+\<Delta\>t|)>
  </equation*>

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|2>>
    <associate|auto-3|<tuple|2.1|2>>
    <associate|auto-4|<tuple|3|3>>
    <associate|auto-5|<tuple|3.1|3>>
    <associate|auto-6|<tuple|3.2|3>>
    <associate|auto-7|<tuple|3.3|3>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Problem
      setting> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Numerical
      Method for NLS> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>2D example
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Numerical
      Example> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>

      <with|par-left|<quote|1tab>|3.1<space|2spc>1D
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1tab>|3.2<space|2spc>2D
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|3.3<space|2spc>Post-process
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>
    </associate>
  </collection>
</auxiliary>