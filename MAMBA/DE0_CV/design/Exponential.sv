module Exponential 
    #(
    parameter SIZE = 32
    ) 
    (
    input logic clk,
    input logic rst,
    input logic start,
    input logic [SIZE-1:0] data,

    output logic [SIZE-1:0] out_data,
    output logic finish,
    
    );
    
    //**********************
    //      data * 369
    //**********************
    logic [SIZE+8:0] data_369;
    assign data_369 = data * 369;

    //**********************
    //         >> 8
    //**********************
    logic [SIZE:0] shift_8_data;
    assign shift_8_data = data_369 >> 8;

    //**********************
    //          Z
    //**********************
    logic [SIZE-1:0] z;
    assign z = shift_8_data >> 8;

    //**********************
    //          F
    //**********************
    logic [SIZE-1:0] f;
    assign f = z & 32'h0000FFFF;

    //**********************
    //          2^f
    //**********************
    logic [SIZE-1:0] pow_2_f;
    assign pow_2_f = 256 + f;

    assign out_data = pow_2_f;

endmodule

