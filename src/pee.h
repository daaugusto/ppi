/** TAD: PEE **/

/** Tipo exportado **/
typedef struct t_individual Individual;

/** Funcoes exportadas **/
/** ****************************************************************************************** **/
/** ************************************* Function init ************************************** **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void init( double** input, double** model, double* obs, int nlin, int argc, char **argv );

/** ****************************************************************************************** **/
/** *********************************** Function evaluate ************************************ **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void evaluate( Individual* individual );

/** ****************************************************************************************** **/
/** ******************************* Function individual_print ******************************** **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void individual_print( const Individual* individual, FILE* out );

/** ****************************************************************************************** **/
/** ****************************** Function generate_population ****************************** **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void generate_population( Individual* population );

/** ****************************************************************************************** **/
/** *********************************** Function crossover *********************************** **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void crossover( const int* father, const int* mother, int* offspring1, int* offspring2 );
   
/** ****************************************************************************************** **/
/** *********************************** Function mutation ************************************ **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void mutation( int* genome );

/** ****************************************************************************************** **/
/** ************************************* Function clone ************************************* **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void clone( const Individual* original, Individual* clone );

/** ****************************************************************************************** **/
/** *********************************** Function tournament ********************************** **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
const Individual* tournament( const Individual* population );

/** ****************************************************************************************** **/
/** ************************************* Function evolve ************************************ **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
Individual evolve ();
