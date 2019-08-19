Search.setIndex({docnames:["fem","index","mesh","misc","modules","solvers"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.todo":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["fem.rst","index.rst","mesh.rst","misc.rst","modules.rst","solvers.rst"],objects:{"":{fem:[0,0,0,"-"],mesh:[2,0,0,"-"],misc:[3,0,0,"-"],solvers:[5,0,0,"-"]},"fem.base_fspace":{BaseFunctionSpace:[0,1,1,""]},"fem.base_fspace.BaseFunctionSpace":{function_spaces:[0,2,1,""]},"fem.base_weakform":{BaseWeakForm:[0,1,1,""]},"fem.struct_templates":{AcousticConstants:[0,1,1,""],Space:[0,1,1,""]},"fem.struct_templates.AcousticConstants":{Pe:[0,2,1,""],Re:[0,2,1,""],gamma:[0,2,1,""]},"fem.struct_templates.Space":{dimension:[0,2,1,""],element_type:[0,2,1,""],order:[0,2,1,""]},"fem.tv_acoustic_fspace":{ComplexTVAcousticFunctionSpace:[0,1,1,""],DEFAULT_TVACOUSTIC_SPACES:[0,3,1,""],TVAcousticFunctionSpace:[0,1,1,""]},"fem.tv_acoustic_fspace.ComplexTVAcousticFunctionSpace":{pressure_function_space:[0,2,1,""],temperature_function_space:[0,2,1,""],velocity_function_space:[0,2,1,""]},"fem.tv_acoustic_fspace.TVAcousticFunctionSpace":{pressure_function_space:[0,2,1,""],temperature_function_space:[0,2,1,""],velocity_function_space:[0,2,1,""]},"fem.tv_acoustic_weakform":{BaseTVAcousticWeakForm:[0,1,1,""],ComplexTVAcousticWeakForm:[0,1,1,""],TVAcousticWeakForm:[0,1,1,""],parse_trialtest:[0,3,1,""]},"fem.tv_acoustic_weakform.BaseTVAcousticWeakForm":{allowed_stress_bcs:[0,4,1,""],allowed_temperature_bcs:[0,4,1,""],boundary_components:[0,2,1,""],density:[0,2,1,""],dirichlet_boundary_conditions:[0,2,1,""],entropy:[0,2,1,""],function_space:[0,2,1,""],function_space_factory:[0,2,1,""],get_constants:[0,2,1,""],heat_flux:[0,2,1,""],pressure_function_space:[0,2,1,""],shear_stress:[0,2,1,""],spatial_component:[0,2,1,""],stress:[0,2,1,""],temperature_function_space:[0,2,1,""],temporal_component:[0,2,1,""],velocity_function_space:[0,2,1,""]},"fem.tv_acoustic_weakform.ComplexTVAcousticWeakForm":{boundary_components:[0,2,1,""],complex_component:[0,2,1,""],function_space_factory:[0,4,1,""],real_function_space:[0,2,1,""],spatial_component:[0,2,1,""],temporal_component:[0,2,1,""]},"fem.tv_acoustic_weakform.TVAcousticWeakForm":{boundary_components:[0,2,1,""],energy:[0,2,1,""],function_space_factory:[0,4,1,""],spatial_component:[0,2,1,""],temporal_component:[0,2,1,""]},"mesh.boundaryelement":{BSplineElement:[2,1,1,""],BoundaryElement:[2,1,1,""],LineElement:[2,1,1,""],on_surface:[2,3,1,""]},"mesh.boundaryelement.BSplineElement":{generate_boundary:[2,2,1,""],get_curvature:[2,2,1,""],get_displacement:[2,2,1,""],get_normal:[2,2,1,""]},"mesh.boundaryelement.BoundaryElement":{control_points:[2,2,1,""],estimate_points_number:[2,2,1,""],generate_boundary:[2,2,1,""],get_curvature:[2,2,1,""],get_displacement:[2,2,1,""],get_normal:[2,2,1,""],surface_index:[2,4,1,""]},"mesh.geometry":{Geometry:[2,1,1,""],SimpleDomain:[2,1,1,""],newfolder:[2,3,1,""]},"mesh.geometry.Geometry":{boundary_parts:[2,2,1,""],compile_mesh:[2,2,1,""],ds:[2,2,1,""],dx:[2,2,1,""],geo_to_mesh:[2,2,1,""],get_boundary_measure:[2,2,1,""],load_mesh:[2,2,1,""],mark_boundaries:[2,2,1,""],n:[2,2,1,""],write_geom:[2,2,1,""]},"misc.optimization_mixin":{OptimizationMixin:[3,1,1,""]},"misc.optimization_mixin.OptimizationMixin":{jacobian:[3,2,1,""],minimize:[3,2,1,""],objective:[3,2,1,""]},"misc.time_storage":{TimeGridError:[3,5,1,""],TimeSeries:[3,1,1,""]},"misc.time_storage.TimeSeries":{apply:[3,2,1,""],first:[3,2,1,""],from_dict:[3,2,1,""],from_list:[3,2,1,""],integrate:[3,2,1,""],interpolate_to_keys:[3,2,1,""],last:[3,2,1,""],values:[3,2,1,""]},"misc.type_checker":{is_dolfin_exp:[3,3,1,""],is_numeric_argument:[3,3,1,""],is_numeric_tuple:[3,3,1,""]},"solvers.base_solver":{BaseSolver:[5,1,1,""],EigenvalueSolver:[5,1,1,""]},"solvers.base_solver.BaseSolver":{output_field:[5,2,1,""],solve:[5,2,1,""],visualization_files:[5,2,1,""]},"solvers.base_solver.EigenvalueSolver":{configure_solver:[5,2,1,""],retrieve_eigenpair:[5,2,1,""],set_solver_operators:[5,2,1,""],solve:[5,2,1,""]},"solvers.eigenvalue_tv_acoustic_solver":{EigenvalueTVAcousticSolver:[5,1,1,""]},"solvers.eigenvalue_tv_acoustic_solver.EigenvalueTVAcousticSolver":{extract_solution:[5,2,1,""],lhs:[5,2,1,""],reconstruct_eigenpair:[5,2,1,""],rhs:[5,2,1,""]},"solvers.unsteady_tv_acoustic_solver":{UnsteadyTVAcousticSolver:[5,1,1,""]},"solvers.unsteady_tv_acoustic_solver.UnsteadyTVAcousticSolver":{initial_state:[5,2,1,""],initialize_solver:[5,2,1,""],solve:[5,2,1,""],solve_adjoint:[5,2,1,""],solve_direct:[5,2,1,""],visualization_files:[5,2,1,""]},fem:{base_fspace:[0,0,0,"-"],base_weakform:[0,0,0,"-"],struct_templates:[0,0,0,"-"],tv_acoustic_fspace:[0,0,0,"-"],tv_acoustic_weakform:[0,0,0,"-"]},mesh:{boundaryelement:[2,0,0,"-"],geometry:[2,0,0,"-"]},misc:{optimization_mixin:[3,0,0,"-"],time_storage:[3,0,0,"-"],type_checker:[3,0,0,"-"]},solvers:{base_solver:[5,0,0,"-"],eigenvalue_tv_acoustic_solver:[5,0,0,"-"],unsteady_tv_acoustic_solver:[5,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"],"5":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute","5":"py:exception"},terms:{"abstract":[0,2,5],"case":0,"class":[0,2,3,5],"default":[0,2],"final":5,"float":5,"function":[0,3],"int":5,"new":[2,3],"return":[2,3,5],"static":[0,2],"true":[0,3,5],"while":5,LHS:5,RHS:5,The:[2,3,5],Then:5,__dolfin__:1,__firecrest__:1,__pysplines__:1,_jacobian:3,_object:3,_objective_st:3,abc:[0,2,3,5],account:2,acoust:[0,5],acousticconst:0,activ:1,actual:5,addit:1,adiabat:[0,2],adjoint:[3,5],after:5,alia:0,all:[0,2,5],allowed_stress_bc:0,allowed_temperature_bc:0,appear:5,appendix:5,appli:3,appropri:0,arc:2,arg:[3,5],argument:0,arrai:3,artstat:1,assembl:5,assign:2,associ:2,attribut:2,back:5,backsubstitut:5,backward:5,base:[0,2,3,5],base_fspac:4,base_solv:4,base_weakform:4,basefunctionspac:0,basesolv:5,basetvacousticweakform:0,baseweakform:0,bcond:[0,2],bcs:5,between:5,bnd:3,bool:5,boundari:[0,2],boundary_compon:0,boundary_el:2,boundary_part:2,boundary_typ:2,boundaryel:4,bring:5,bsplineel:2,c_p:0,c_v:0,cach:[3,5],calcul:5,can:[2,5],capac:0,certain:2,characterist:2,chosen:0,circular:2,classmethod:3,clone:1,code:2,collect:[0,2,3],com:1,command:2,compar:5,compil:2,compile_mesh:2,complex:[0,5],complex_compon:0,complex_flag:0,complex_shift:5,complextvacousticfunctionspac:0,complextvacousticweakform:0,compon:[0,5],comput:3,conda:1,condit:[2,5],conduct:0,configur:5,configure_solv:5,consist:[2,3],constant:[0,5],construct:5,content:4,control:[2,3,5],control_point:2,convert:2,corwinpro:1,cp_index:2,crank:5,crank_nicolson:5,creat:[1,2,3],current:3,data:3,deactiv:1,default_tvacoustic_spac:0,defin:[0,5],demo:5,demo_elastodynam:5,densiti:0,dev:5,dict:3,dictionari:2,differ:[3,5],dim:2,dimens:[0,2],dimension:2,direct:[3,5],directori:2,dirichlet:0,dirichlet_boundary_condit:0,dirichletbc:0,discret:5,discuss:5,doc:5,dolf:2,dolf_fil:2,dolfin:[0,1,2,5],domain:[0,2,5],done:5,doubl:5,each:5,effici:5,eigenmod:5,eigensolv:5,eigenvalu:5,eigenvalue_toler:5,eigenvalue_tv_acoustic_solv:4,eigenvaluesolv:5,eigenvaluetvacousticsolv:5,el_siz:2,elastodynam:5,element:[0,2],element_typ:0,energi:0,entiti:2,entropi:0,environ:1,equat:0,estimate_points_numb:2,except:3,expect:0,extens:2,extract_solut:5,factor:5,factori:0,fals:[0,5],fedor:1,fem:4,fenic:1,fenicsproject:[1,5],field:[0,5],file:2,finit:0,firecrest:[0,5],firecrest_emblem3002:1,first:[2,3,5],flux:0,follow:3,forc:0,forg:1,form:[0,2,5],format:2,free:0,from:[0,1,2,3,5],from_dict:3,from_list:3,full:2,func:3,function_spac:0,function_space_factori:0,gamma:0,gener:[0,2,5],generate_boundari:2,geo:2,geo_fil:2,geo_to_mesh:2,geom:2,geometr:[0,2],geometri:4,get_boundary_measur:2,get_const:0,get_curvatur:2,get_displac:2,get_norm:2,ghost:5,git:1,github:1,given:3,gradient:3,grid:3,hand:5,handl:5,heat:0,heat_flux:[0,2],here:1,highest:5,histori:5,html:5,http:[1,5],imaginari:5,img:1,imped:0,implement:3,increment:5,index:[1,2,5],individu:2,inflow:0,initi:5,initial_st:5,initialize_solv:5,instal:1,instanc:3,instead:5,integr:3,intermedi:3,interpolate_to_kei:3,is_dolfin_exp:3,is_linearis:0,is_numeric_argu:3,is_numeric_tupl:3,isotherm:0,jacobian:3,kei:2,keys_seri:3,kwarg:[0,2,3,5],last:3,length:5,level:5,lhs:5,like:0,line:2,linear:5,lineel:2,list:[2,3],load_mesh:2,logo:1,lusolv:5,mark:2,mark_boundari:2,markers_dict:2,market_dict:2,matplotlib:1,matric:5,matrix:5,measur:2,mesh:[0,4],mesh_fold:2,mesh_format:2,mesh_nam:2,meshfunct:2,meshio:1,messag:3,method:[2,3],mid_point:3,minim:3,misc:4,mixin:3,modifi:5,modul:[1,4],momentum:0,more:5,msh:2,msh_file:2,msh_format:2,much:5,mump:5,must:5,name:[2,5],necessari:[0,1,5],neither:2,neumann:0,newfold:2,nicolson:5,nof_converg:5,non:5,none:[0,2,3,5],norm:5,normal:[2,5],normal_forc:0,normal_veloc:0,noslip:[0,2],noslip_boundary_1:2,noslip_boundary_2:2,number:[0,5],object:[0,2,3,5],obtain:5,on_surfac:2,onc:5,one:[3,5],onli:5,optim:3,optimization_mixin:4,optimizationmixin:3,order:0,ordereddict:3,org:5,output:5,output_field:5,packag:[1,4],page:1,pair:[2,5],paper:5,param:[0,2,5],paramet:[3,5],parametr:2,pars:0,parse_trialtest:0,part:5,pass:5,path:2,pecklet:0,perform:5,petsc:5,pg_:2,pg_geometri:2,physic:2,pick:0,piec:2,pip:1,png:1,point:2,polynomi:0,pressur:0,pressure_function_spac:0,problem:[0,5],properti:[0,2,3,5],provid:[2,3],pygmsh:2,pysplin:1,python:5,quietvictori:1,rang:5,ratio:0,read:2,readabl:2,real:[0,5],real_function_spac:0,recombin:5,reconstruct_eigenpair:5,report:5,repres:2,represent:2,requir:3,result:3,retrieve_eigenpair:5,reus:5,reynold:0,rhs:5,right:5,robin:0,routin:2,same:5,save:3,scalar:0,search:1,second:2,see:5,self:2,seri:3,set:[1,2,5],set_solver_oper:5,shear_stress:0,should:0,side:5,simpledomain:2,sinc:5,size:2,slepc:5,slip:[0,2],snapshot:3,solut:5,solv:5,solve_adjoint:5,solve_direct:5,solver:[3,4],solver_typ:5,someth:0,sourc:[0,1,2,3,5],space:[0,5],spatial:[0,5],spatial_compon:0,specif:2,specifi:2,spline:2,src:1,stamp:3,start_tim:3,state:[0,3,5],step:[3,5],storag:3,store:[2,3,5],str:2,stress:0,struct_templ:4,submodul:4,sup:1,surfac:2,surface_index:2,surface_lin:2,symmetri:5,system:5,temperatur:0,temperature_function_spac:0,templat:[0,3],template_grid:3,tempor:[0,5],temporal_compon:0,test:0,them:5,therefor:5,thermal:0,thermal_accommod:0,thermovisc:[0,5],thi:[2,5],time:[3,5],time_schem:5,time_storag:4,timegriderror:3,timeseri:3,titov:1,toler:5,track:2,trial:0,tupl:[0,5],tv_acoustic_fspac:4,tv_acoustic_weakform:4,tvacoust:0,tvacousticfunctionspac:0,tvacousticweakform:0,type:[0,2,5],type_check:4,uniqu:2,unord:3,unsteadi:5,unsteady_tv_acoustic_solv:4,unsteadytvacousticsolv:5,usag:0,used:3,user:1,using:2,valu:[2,3,5],vari:5,vector:[0,3,5],veloc:0,velocity_function_spac:0,verbos:5,verifi:5,view:3,viscos:0,visualization_fil:5,weak:0,when:2,which:5,whole:2,width:1,write:2,write_geom:2,www:1,xdmf:2,xml:2,year:5,zero:5},titles:["fem package","Welcome to firecrest's documentation!","mesh package","misc package","firecrest","solvers package"],titleterms:{base_fspac:0,base_solv:5,base_weakform:0,boundaryel:2,content:[0,2,3,5],document:1,eigenvalue_tv_acoustic_solv:5,fem:0,firecrest:[1,4],geometri:2,indic:1,mesh:2,misc:3,modul:[0,2,3,5],optimization_mixin:3,packag:[0,2,3,5],solver:5,struct_templ:0,submodul:[0,2,3,5],tabl:1,time_storag:3,tv_acoustic_fspac:0,tv_acoustic_weakform:0,type_check:3,unsteady_tv_acoustic_solv:5,welcom:1}})