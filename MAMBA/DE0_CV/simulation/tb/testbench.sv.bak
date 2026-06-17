module testbench;


	logic [3:0]a,b,s;
	

adder_4bit a1(
	.a(a),
	.b(b),
	.s(s)
	);

	
	initial begin

		#10 a = 5; 		b = 3;
		#10 a = 1; 		b = 1;
		#10 a = 2; 		b = 8;
		#10 a = 7; 		b = 7;
		#10 a = 6; 		b = 6;
		#1000 $stop;
	end
endmodule