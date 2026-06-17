module Discretization
	#(
		parameter A_size = 16,
		parameter B_size = 16,
		parameter Delta_size = 16,
		parameter L = 4,
		parameter D_IN = 32,
		parameter N = 16
	)
	(	
		input logic clk,
		input logic rst,
		input logic start,
		input logic [0:15] data,
		
		// output等等寫
		output logic finish
		
	);
	//////////////////////////////////////////////////////  INPUT
	
	// A
	logic start_a;
	logic [0:A_size-1] reg_A [0:D_IN-1][0:N-1];
	
	//	B
	logic start_b;
	logic [0:B_size-1] reg_B [0:L-1][0:N-1];
	
	//	DELTA
	logic start_delta;
	logic [0:Delta_size-1] reg_delta [0:L-1][0:D_IN-1];
	
	// COUNTER
	logic data_cnt_rst;
	logic [0:10] data_cnt;
	
	// IN_FSM
	typedef enum {IDLE, START, A, B, DELTA, FINISH} input_state;
	input_state in_ps, in_ns;
	
	//**********************
	//			COUNTER
	//**********************
	always_ff @(posedge clk) begin
		if (rst | data_cnt_rst) begin
			data_cnt <= 0;
		end else begin
			data_cnt <= data_cnt + 1;
		end
	end
	
	//**********************
	//			A_input  (d_in,n)
	//**********************
	always_ff @(posedge clk) begin
		if (start_a) begin
			reg_A[data_cnt%D_IN][data_cnt/N] <= data[0:A_size-1];
		end
	end
	
	//**********************
	//			B_input  (l,n)
	//**********************
	always_ff @(posedge clk) begin
		if (start_b) begin
			reg_B[data_cnt%L][data_cnt/L] <= data[0:B_size-1];
		end
	end
	
	//*************************
	//			delta_input  (l_d_in)
	//*************************
	always_ff @(posedge clk) begin
		if (start_delta) begin
			reg_delta[data_cnt%L][data_cnt/L] <= data[0:Delta_size-1];
		end
	end
	
	
	//******************
	//			IN_FSM
	//******************
	always_ff @(posedge clk) begin
		if (rst) begin
			in_ps <= IDLE;
		end else begin
			in_ps <= in_ns;
		end
	end
	
	always_comb begin
		
		data_cnt_rst	= 0;
		start_a			= 0;
		start_b			= 0;
		start_delta		= 0;
		finish			= 0;
		in_ns 			= in_ps;
		
		case (in_ps)
			IDLE:	begin
				in_ns = START;
			end
			START: begin
				if (start) begin
					data_cnt_rst = 1;
					in_ns = A;
				end
			end
			A: begin
				start_a = 1;
				if (data_cnt >= D_IN*N) begin
					data_cnt_rst = 1;
					in_ns = B;
				end
			end
			B: begin
				start_b = 1;
				if (data_cnt >= L*N) begin
					data_cnt_rst = 1;
					in_ns = DELTA;
				end
			end
			DELTA: begin
				start_delta = 1;
				if (data_cnt >= L*D_IN) begin
					data_cnt_rst = 1;
					in_ns = FINISH;
				end
				
			end
			FINISH: begin
				finish = 1;
				in_ns = IDLE;
			end
		endcase
		
	end
	//////////////////////////////////////////////////////	MULTIPLIER
	
	// delta_A
	logic [0:Delta_size*A_size] delta_A [0:D_IN-1][0:N-1];
	
	//******************
	//		delta_A
	//******************
	
		
		
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	//////////////////////////////////////////////////////
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
endmodule
